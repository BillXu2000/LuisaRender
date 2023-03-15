#include "core/basic_types.h"
#include <util/rng.h>
#include <base/pipeline.h>
#include <base/integrator.h>

#include <util/progress_bar.h>
#include <base/display.h>
#include <core/mathematics.h>

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
            camera->film()->accumulate(pixel_id, shutter_weight * L);
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
        SampledSpectrum beta{swl.dimension(), cs.weight};

        auto ray = cs.ray;

        // auto pixel = camera->get_pixel(ray->direction(), time, u_filter, u_lens);

        // return (pixel[0], pixel[1], 0);
        // return (make_float2(pixel)[0] / 2000, make_float2(pixel)[0] / 2000, 0);
        // auto ans = Float3{0, 0, 0};
        // $if(pixel[0] > pixel_id[0]) {
        //     ans[0] = 1;
        // }
        // $else {
        //     ans[1] = 1;
        // };
        // return ans;
        // return {pixel[0] - pixel_id[0], 0, 0};
        // return {camera->filter()->sample(u_filter).offset[0], 0, 0};
        // auto offset = camera->filter()->sample(u_filter).offset;
        // auto pixel = camera->get_pixel(ray->direction(), time, make_float2(), u_lens);
        // Float3 ans{offset[0], offset[1], 0};
        // Float3 ans{0, 0, 0};
        // $if (pixel[0] == pixel_id[0]) {
        //     ans[2] = 1;
        // };
        // $if (pixel[0] != pixel_id[0]) {
        //     ans[2] = -1;
        // };
        // $if (offset[0] >= 0) {
        //     ans[0] = 1;
        // };
        // $if (offset[0] <= 1 & offset[0] >= -1) {
        //     ans[0] = 1;
        // };
        // $if (offset[0] > 1 | offset[0] < -0.5f) {
        //     ans[1] = 1;
        // };
        // $if (offset[0] == 0) {
        //     ans[2] = 1;
        // };
        // return ans;


        $loop {

            // trace
            auto wo = -ray->direction();
            auto it = pipeline().geometry()->intersect(ray);

            // compute direct lighting
            $if(!it->shape().has_surface()) { $break; };

            auto light_sample = LightSampler::Sample::zero(swl.dimension());
            auto occluded = def(false);

            // sample one light
            auto u_light_selection = sampler()->generate_1d();
            auto u_light_surface = sampler()->generate_2d();
            auto u_w = sampler()->generate_2d();
            light_sample = light_sampler()->sample_ray(
                u_light_selection, u_light_surface, u_w, swl, time);
            auto it_light = pipeline().geometry()->intersect(light_sample.shadow_ray);
            // auto ray_connect = it_light->spawn_ray_to(it->p());
            auto ray_connect = it_light->spawn_ray_to(ray->origin());
            occluded = pipeline().geometry()->intersect_any(ray_connect);

            // // trace shadow ray
            // $if(light_sample.eval.pdf > 0.f &
            //     light_sample.eval.L.any([](auto x) { return x > 0.f; })) {
            //     occluded = pipeline().geometry()->intersect_any(light_sample.shadow_ray);
            // };
            auto surface_tag = it->shape().surface_tag();
            pipeline().surfaces().dispatch(surface_tag, [&](auto surface) noexcept {
                auto closure = surface->closure(it, swl, wo, 1.f, time);
                auto wi = -ray_connect->direction();
                auto eval = closure->evaluate(wo, wi);
                beta *= eval.f;
            });
            $if(light_sample.eval.pdf > 0.0f & !occluded) {
                auto pixel = camera->get_pixel(-ray_connect->direction(), time, make_float2(), u_lens);
                $if (pixel[0] < 32768) {
                    camera->film()->accumulate(pixel, shutter_weight * make_float3(1));
                };
                // Li += cs.weight * light_sample.eval.L / light_sample.eval.pdf * beta;
                // auto pixel = camera->get_pixel(ray->direction(), time, offset, u_lens);
                // Li += cs.weight * light_sample.eval.L;
            };
            $break;
        };
        return make_float3();
        // return spectrum->srgb(swl, Li);
    }
};

luisa::unique_ptr<Integrator::Instance> VCM::build(
    Pipeline &pipeline, CommandBuffer &cb) const noexcept {
    return luisa::make_unique<VCMInstance>(pipeline, cb, this);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::VCM)
