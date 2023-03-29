#include "core/basic_types.h"
#include "util/spec.h"
#include <util/rng.h>
#include <base/pipeline.h>
#include <base/integrator.h>

#include <util/progress_bar.h>
#include <base/display.h>
#include <core/mathematics.h>

namespace {
    float fov_area;
}

namespace luisa::render {

class VCM final : public ProgressiveIntegrator {

private:
    bool _remap;
    bool _shading;

public:
    VCM(Scene *scene, const SceneNodeDesc *desc) noexcept
        : ProgressiveIntegrator{scene, desc},
          _remap{desc->property_bool_or_default("remap", true)},
          _shading{desc->property_bool_or_default("shading", true)} {}
    [[nodiscard]] auto remap() const noexcept { return _remap; }
    [[nodiscard]] auto shading() const noexcept { return _shading; }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] luisa::unique_ptr<Integrator::Instance> build(
        Pipeline &pipeline, CommandBuffer &cb) const noexcept override;
};

class VCMInstance final : public ProgressiveIntegrator::Instance {

public:
    using ProgressiveIntegrator::Instance::Instance;

protected:
    void _render_one_camera(CommandBuffer &command_buffer, Camera::Instance *camera) noexcept override {
        auto spp = camera->node()->spp();
        auto resolution = camera->film()->node()->resolution();
        auto image_file = camera->node()->file();

        auto pixel_count = resolution.x * resolution.y;
        sampler()->reset(command_buffer, resolution, pixel_count, spp);
        command_buffer << compute::synchronize();

        LUISA_INFO(
            "Rendering to '{}' of resolution {}x{} at {}spp.",
            image_file.string(),
            resolution.x, resolution.y, spp);

        using namespace luisa::compute;

        Kernel2D render_kernel = [&](UInt frame_index, Float time, Float shutter_weight) noexcept {
            set_block_size(16u, 16u, 1u);
            auto pixel_id = dispatch_id().xy();
            auto L = Li_vcm(camera, frame_index, pixel_id, time, shutter_weight);
            camera->film()->accumulate(pixel_id, make_float3(0));
        };

        Clock clock_compile;
        auto render = pipeline().device().compile(render_kernel);
        auto integrator_shader_compilation_time = clock_compile.toc();
        LUISA_INFO("Integrator shader compile in {} ms.", integrator_shader_compilation_time);
        auto shutter_samples = camera->node()->shutter_samples();
        command_buffer << synchronize();

        LUISA_INFO("Rendering started.");
        Clock clock;
        ProgressBar progress;
        progress.update(0.);
        auto dispatch_count = 0u;
        auto sample_id = 0u;
        for (auto s : shutter_samples) {
            pipeline().update(command_buffer, s.point.time);
            for (auto i = 0u; i < s.spp; i++) {
                command_buffer << render(sample_id++, s.point.time, s.point.weight)
                                    .dispatch(resolution);
                auto dispatches_per_commit =
                    display()->should_close() ?
                        node<ProgressiveIntegrator>()->display_interval() :
                        32u;
                if (++dispatch_count % dispatches_per_commit == 0u) [[unlikely]] {
                    dispatch_count = 0u;
                    auto p = sample_id / static_cast<double>(spp);
                    if (display()->update(command_buffer, sample_id)) {
                        progress.update(p);
                    } else {
                        command_buffer << [&progress, p] { progress.update(p); };
                    }
                }
            }
        }
        command_buffer << synchronize();
        progress.done();

        auto render_time = clock.toc();
        LUISA_INFO("Rendering finished in {} ms.", render_time);
    }

    [[nodiscard]] Float3 Li(const Camera::Instance *camera, Expr<uint> frame_index,
                            Expr<uint2> pixel_id, Expr<float> time) const noexcept override {
        return Li_vcm(camera, frame_index, pixel_id, time, 1.f);
    }
    [[nodiscard]] Float3 Li_vcm(const Camera::Instance *camera, Expr<uint> frame_index,
                            Expr<uint2> pixel_id, Expr<float> time, Expr<float> shutter_weight) const noexcept {
        sampler()->start(pixel_id, frame_index);
        auto u_filter = sampler()->generate_pixel_2d();
        auto u_lens = camera->node()->requires_lens_sampling() ? sampler()->generate_2d() : make_float2(.5f);
        auto cs = camera->generate_ray(pixel_id, time, u_filter, u_lens);
        auto spectrum = pipeline().spectrum();
        auto swl = spectrum->sample(spectrum->node()->is_fixed() ? 0.f : sampler()->generate_1d());
        SampledSpectrum Li{swl.dimension(), 0.f};
        SampledSpectrum beta_light{swl.dimension(), shutter_weight};

        auto ray_camera = cs.ray;

        auto ans = def(make_float3());
        // $if (cs.weight >= 1) {
        //     ans[0] = 1;
        // };
        // $if (cs.weight < 1 & cs.weight >= 0) {
        //     ans[1] = 1;
        // };
        // $if (cs.weight < 0) {
        //     ans[2] = 1;
        // };
        // camera->film()->accumulate(pixel_id, ans);
        // return float3();

        for (int i = 0; i < 1; i++) { // light ray samples
            auto u_light_selection = sampler()->generate_1d();
            auto u_light_surface = sampler()->generate_2d();
            auto u_w = sampler()->generate_2d();
            auto light_sample = light_sampler()->sample_ray(u_light_selection, u_light_surface, u_w, swl, time);
            auto it_light = pipeline().geometry()->intersect(light_sample.shadow_ray);
            auto wi = -light_sample.shadow_ray->direction();
            auto surface_tag = it_light->shape()->surface_tag();
            auto ray_connect = it_light->spawn_ray_to(ray_camera->origin());
            auto occluded = pipeline().geometry()->intersect_any(ray_connect);
            pipeline().surfaces().dispatch(surface_tag, [&](auto surface) noexcept {
                auto wo = ray_connect->direction();
                auto closure = surface->closure(it_light, swl, wo, 1.f, time);
                // auto eval = closure->evaluate(wo, wi);
                auto eval = closure->evaluate(wi, wo);
                $if(!occluded) {
                    auto pixel = camera->get_pixel(-ray_connect->direction(), time, make_float2(), u_lens);
                    $if (pixel[0] < 32768) {
                        auto dist_sqr = distance_squared(it_light->p(), ray_camera->origin());
                        Float pdf = Float((1.f / (2.04973f * 0.636425f) / 4.f)) / abs_dot(it_light->ng(), wo) * dist_sqr;
                        Float light_1_pdf = abs_dot(it_light->ng(), wi) * inv_pi;
                        // camera->film()->accumulate(pixel, spectrum->srgb(swl, k_c * eval.f * light_sample.eval.L / light_sample.eval.pdf / dist_sqr), 0.f);
                        Float3 measure = inv_pi * 0.1f * dot(wi, it_light->ng()) * make_float3(9, 9, 10);
                        float pixel_area = 0.75;
                        Float camera_dot = abs_dot(wo, normalize(make_float3(0.5, -0.5, -1)));
                        Float camera_pdf = pixel_area * camera_dot * camera_dot * camera_dot;
                        camera->film()->accumulate(pixel, measure / pdf / light_1_pdf / camera_pdf, 0.f);
                            // auto pdf = Float((1.f / (2.04973f * 0.636425f) / 4.f)) / abs_dot(it->ng(), wi) * distance_squared(it->p(), it_2->p());
                            // ans = inv_pi * dot(wi, it->ng()) * 0.1f * make_float3(9, 9, 10) / pdf;






                        /*
                        // camera->film()->accumulate(pixel, spectrum->srgb(swl, shutter_weight * beta * eval.f));
                        // auto dist_sqr = distance_squared(light_sample.shadow_ray->origin(), it_light->p());
                        auto dist_sqr = distance_squared(it_light->p(), ray_camera->origin());
                        auto resolution = camera->film()->node()->resolution();
                        // auto camera_rate = 2 * atan(resolution.x / resolution.y * tan(camera.fo))
                        // auto k_c = 2 * pi * float(1 / 1.5708) * beta_light * (1 / dot(-ray_connect->direction(), normalize(make_float3(0.5, -0.5, -1))));
                        auto k_c = 2 * pi * float(1 / 1.5708) * beta_light;
                        camera->film()->accumulate(pixel, spectrum->srgb(swl, k_c * eval.f * light_sample.eval.L / light_sample.eval.pdf / dist_sqr), 0.f);
                        // camera->film()->accumulate(pixel, spectrum->srgb(swl, beta_light * eval.f * light_sample.eval.L / light_sample.eval.pdf / dist_sqr));
                        // camera->film()->accumulate(pixel, spectrum->srgb(swl, beta_light * eval.f));
                        // camera->film()->accumulate(pixel, spectrum->srgb(swl, eval.f));
                        // camera->film()->accumulate(pixel, spectrum->srgb(swl, SampledSpectrum(1.f)));
                        // camera->film()->accumulate(pixel, spectrum->srgb(swl, light_sample.eval.L / light_sample.eval.pdf / dist_sqr));
                        // camera->film()->accumulate(pixel, wo);
                        */
                    };
                };
            });

        }
        return make_float3();
        SampledSpectrum beta{swl.dimension(), cs.weight * shutter_weight};


        $loop {

            // trace
            auto it = pipeline().geometry()->intersect(ray_camera);
            ans = it->p();

            // compute direct lighting
            $if(!it->shape()->has_surface()) { $break; };

            auto light_sample = LightSampler::Sample::zero(swl.dimension());
            auto occluded = def(false);

            // sample one light
            auto u_light_selection = sampler()->generate_1d();
            auto u_light_surface = sampler()->generate_2d();
            auto u_w = sampler()->generate_2d();
            light_sample = light_sampler()->sample_ray(u_light_selection, u_light_surface, u_w, swl, time);
            // light_sample = light_sampler()->sample_ray(u_light_selection, u_light_surface, make_float2(pixel_id) / 2000.f, swl, time);
            auto it_light = pipeline().geometry()->intersect(light_sample.shadow_ray);
            auto wo = -light_sample.shadow_ray->direction();
            // auto ray_connect = it_light->spawn_ray_to(it->p());
            auto ray_connect = it_light->spawn_ray_to(ray_camera->origin());
            occluded = pipeline().geometry()->intersect_any(ray_connect);

            // // trace shadow ray
            // $if(light_sample.eval.pdf > 0.f &
            //     light_sample.eval.L.any([](auto x) { return x > 0.f; })) {
            //     occluded = pipeline().geometry()->intersect_any(light_sample.shadow_ray);
            // };
            auto surface_tag = it->shape()->surface_tag();
            pipeline().surfaces().dispatch(surface_tag, [&](auto surface) noexcept {
                auto closure = surface->closure(it_light, swl, wo, 1.f, time);
                auto wi = ray_connect->direction();
                auto eval = closure->evaluate(wo, wi);
                $if(!occluded) {
                    auto pixel = camera->get_pixel(-ray_connect->direction(), time, make_float2(), u_lens);
                    $if (pixel[0] < 32768) {
                        // camera->film()->accumulate(pixel, spectrum->srgb(swl, shutter_weight * beta * eval.f));
                        camera->film()->accumulate(pixel, spectrum->srgb(swl, eval.f));
                    };
                };
                beta *= eval.f;
            });
            // $if(light_sample.eval.pdf > 0.0f & !occluded) {
            // $if(!occluded) {
            //     auto pixel = camera->get_pixel(-ray_connect->direction(), time, make_float2(), u_lens);
            //     $if (pixel[0] < 32768) {
            //         camera->film()->accumulate(pixel, shutter_weight * make_float3(1));
            //     };
                // Li += cs.weight * light_sample.eval.L / light_sample.eval.pdf * beta;
                // auto pixel = camera->get_pixel(ray->direction(), time, offset, u_lens);
                // Li += cs.weight * light_sample.eval.L;
            // };
            // camera->film()->accumulate(make_uint2(make_float2(light_sample.shadow_ray->direction() * 100.f + 200.f)), make_float3(1));
            $break;
        };
        return make_float3();
        // return ans;
        // return spectrum->srgb(swl, Li);
    }
};

luisa::unique_ptr<Integrator::Instance> VCM::build(
    Pipeline &pipeline, CommandBuffer &cb) const noexcept {
    return luisa::make_unique<VCMInstance>(pipeline, cb, this);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::VCM)
