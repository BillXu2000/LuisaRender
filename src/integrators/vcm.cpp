#include "core/basic_types.h"
#include "dsl/builtin.h"
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
    uint _max_depth;
    uint _rr_depth;
    float _rr_threshold;

public:
    uint lt_depth, debug_depth;
    VCM(Scene *scene, const SceneNodeDesc *desc) noexcept
        : ProgressiveIntegrator{scene, desc},
          _max_depth{std::max(desc->property_uint_or_default("depth", 10u), 1u)},
          _rr_depth{std::max(desc->property_uint_or_default("rr_depth", 0u), 0u)},
          _rr_threshold{std::max(desc->property_float_or_default("rr_threshold", 0.95f), 0.05f)},
          lt_depth{std::max(desc->property_uint_or_default("lt_depth", 5u), 0u)},
          debug_depth{std::max(desc->property_uint_or_default("debug_depth", 0u), 0u)} {}
    [[nodiscard]] auto max_depth() const noexcept { return _max_depth; }
    [[nodiscard]] auto rr_depth() const noexcept { return _rr_depth; }
    [[nodiscard]] auto rr_threshold() const noexcept { return _rr_threshold; }
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

        auto u_light_selection = sampler()->generate_1d();
        auto u_light_surface = sampler()->generate_2d();
        auto u_w = sampler()->generate_2d();
        Float cos_light;
        auto light_sample = light_sampler()->sample_ray(u_light_selection, u_light_surface, u_w, swl, time, cos_light);
        auto ray = light_sample.shadow_ray;
        auto beta = light_sample.eval.L / light_sample.eval.pdf;

        uint n_lt = node<VCM>()->lt_depth;
        auto v_lt = compute::ArrayFloat3<5>();
        Float pd_last = light_sample.eval.pdf / cos_light;
        Float p_w = 1.f;

        $for(depth, n_lt) {
            auto it= pipeline().geometry()->intersect(ray);
            auto wi = -ray->direction();
            p_w /= pd_last; // pd under
            $if(!it->valid()) { $break; };
            $if(!it->shape().has_surface()) { $break; };

            $if (depth == 0) {
                p_w /= (cos_light * abs_dot(it->ng(), ray->direction()) / distance_squared(ray->origin(), it->p())); // G
            };
            // light tracing sample camera
            auto ray_connect = it->spawn_ray_to(cs.ray->origin());
            auto surface_tag = it->shape().surface_tag();
            auto occluded = pipeline().geometry()->intersect_any(ray_connect);
            pipeline().surfaces().dispatch(surface_tag, [&](auto surface) noexcept {
                auto wo = ray_connect->direction();
                auto closure = surface->closure(it, swl, wo, 1.f, time);
                auto eval = closure->evaluate(wi, wo);
                $if(!occluded) {
                    Float importance;
                    auto pixel = camera->get_pixel(-ray_connect->direction(), time, make_float2(), Float2(), importance);
                    $if (pixel[0] < 32768) {
                        auto dist_sqr = distance_squared(it->p(), cs.ray->origin());
                        $if (depth >= node<VCM>()->debug_depth) {
                            Float cos_light = 1.f; // todo: bx2k hack
                            p_w *= (abs_dot(it->ng(), wo) * cos_light / dist_sqr); // G
                            p_w *= (1 / (importance)); // p_1
                            $if (depth > 0) {
                                p_w *= closure->evaluate(wo, wi).pdf / abs_dot(it->ng(), wi);
                            };
                            Float sqr_heuristic = 1 / (1 + sqr(p_w));
                            camera->film()->accumulate(pixel, spectrum->srgb(swl, sqr_heuristic * beta * eval.f * importance / dist_sqr), 0.f);
                        };
                    };
                };
            });

            auto u_lobe = sampler()->generate_1d();
            auto u_bsdf = sampler()->generate_2d();
            // evaluate material
            auto eta_scale = def(1.f);
            pipeline().surfaces().dispatch(surface_tag, [&](auto surface) noexcept {
                // create closure
                auto closure = surface->closure(it, swl, wi, 1.f, time);

                // apply opacity map
                auto alpha_skip = def(false);
                if (auto o = closure->opacity()) {
                    auto opacity = saturate(*o);
                    alpha_skip = u_lobe >= opacity;
                    u_lobe = ite(alpha_skip, (u_lobe - opacity) / (1.f - opacity), u_lobe / opacity);
                }

                $if(alpha_skip) {
                    ray = it->spawn_ray(ray->direction());
                    // pdf_bsdf = 1e16f;
                }
                $else {
                    if (auto dispersive = closure->is_dispersive()) {
                        $if(*dispersive) { swl.terminate_secondary(); };
                    }
                    // sample material
                    auto surface_sample = closure->sample(wi, u_lobe, u_bsdf);
                    ray = it->spawn_ray(surface_sample.wi);
                    // pdf_bsdf = surface_sample.eval.pdf;
                    auto w = ite(surface_sample.eval.pdf > 0.f, 1.f / surface_sample.eval.pdf, 0.f);
                    beta *= w * surface_sample.eval.f;
                    // apply eta scale
                    auto eta = closure->eta().value_or(1.f);
                    $switch(surface_sample.event) {
                        $case(Surface::event_enter) { eta_scale = sqr(eta); };
                        $case(Surface::event_exit) { eta_scale = sqr(1.f / eta); };
                    };
                    $if (depth > 0) {
                        p_w *= closure->evaluate(surface_sample.wi, wi).pdf / abs_dot(it->ng(), wi);
                    };
                };
            });
            beta = zero_if_any_nan(beta);
            $if(beta.all([](auto b) noexcept { return b <= 0.f; })) { $break; };

            auto u_rr = def(0.f);
            auto rr_depth = node<VCM>()->rr_depth();
            $if (depth + 1u >= rr_depth) { u_rr = sampler()->generate_1d(); };
            auto rr_threshold = node<VCM>()->rr_threshold();
            auto q = max(beta.max() * eta_scale, .05f);
            $if(depth + 1u >= rr_depth) {
                $if(q < rr_threshold & u_rr >= q) { $break; };
                beta *= ite(q < rr_threshold, 1.0f / q, 1.f);
            };
        };
        return make_float3();
    }
};

luisa::unique_ptr<Integrator::Instance> VCM::build(
    Pipeline &pipeline, CommandBuffer &cb) const noexcept {
    return luisa::make_unique<VCMInstance>(pipeline, cb, this);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::VCM)
