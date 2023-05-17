//
// Created by Hercier on 2023/3/4.
//

#include <util/sampling.h>
#include <util/medium_tracker.h>
#include <util/progress_bar.h>
#include <base/pipeline.h>
#include <base/integrator.h>
#include <base/display.h>

namespace luisa::render {

namespace {
    float fov_area;
    float a_pk = 1;
    float a_p0 = 1;
    float a_pi = -1;
    // float r_pm;
    Float inv_r(const Float &k) {
        return ite(k > 0, 1.0f / k, 0.f);
    }

    Float bx_power(const Float &a) {
        return sqr(a);
        // return a;
    }

    Float bx_heuristic(const Float &a, const Float &b) {
        return a * inv_r(a + b);
        // return sqr(a) * inv_r(sqr(a) + sqr(b));
    }
}

using namespace compute;
/* Procedure :
    1.emit photons and save them
    2.(first time only) initialize pixelinfo and get the proper initial radius based on emitted photons
    3.put photons in the hashmap grids
    4.render direct light seperately, stop at high roughness, find nearby 3*3*3 grids for photons and save the informations
    5.using shared(SPPM)/PPM update procedure for pixels
    6.if shared, a seperate update is performed, and the grid_len is also updated according to radius
    7.repeat until end, then draw the indirect light to film
*/
class bxpm final : public ProgressiveIntegrator {

private:
    uint _max_depth;
    uint _rr_depth;
    float _rr_threshold;
    uint _photon_per_iter;
    float _initial_radius;
    bool _separate_direct;
    bool _shared_radius;

public:
    bool enable_lt, enable_rt, enable_pm, enable_i0, enable_i1;
    uint lt_depth, debug_depth;
    bxpm(Scene *scene, const SceneNodeDesc *desc) noexcept
        : ProgressiveIntegrator{scene, desc},
          _max_depth{std::max(desc->property_uint_or_default("depth", 10u), 1u)},
          _rr_depth{std::max(desc->property_uint_or_default("rr_depth", 2u), 0u)},
          _rr_threshold{std::max(desc->property_float_or_default("rr_threshold", 0.95f), 0.05f)},
          _initial_radius{std::max(desc->property_float_or_default("initial_radius", -200.f), -10000.f)},//<0 for world_size/-radius (-grid count)
          _photon_per_iter{std::max(desc->property_uint_or_default("photon_per_iter", 200000u), 10u)},
          _separate_direct{true},                                                  //when false, use photon mapping for all flux and gathering at first intersection. Just for debug
          _shared_radius{desc->property_bool_or_default("shared_radius", true)},//whether or not use the shared radius trick in SPPM paper. True is better in performance.
          enable_lt{desc->property_bool_or_default("enable_lt", true)},
          enable_rt{desc->property_bool_or_default("enable_rt", true)},
          enable_pm{desc->property_bool_or_default("enable_pm", true)},
          enable_i0{desc->property_bool_or_default("enable_i0", true)},
          enable_i1{desc->property_bool_or_default("enable_i1", true)},
          lt_depth{std::max(desc->property_uint_or_default("lt_depth", 5u), 0u)},
          debug_depth{std::max(desc->property_uint_or_default("debug_depth", 0u), 0u)} {}

    [[nodiscard]] auto max_depth() const noexcept { return _max_depth; }
    [[nodiscard]] auto photon_per_iter() const noexcept { return _photon_per_iter; }
    [[nodiscard]] auto rr_depth() const noexcept { return _rr_depth; }
    [[nodiscard]] auto initial_radius() const noexcept { return _initial_radius; }
    [[nodiscard]] auto rr_threshold() const noexcept { return _rr_threshold; }
    [[nodiscard]] auto separate_direct() const noexcept { return _separate_direct; }
    [[nodiscard]] auto shared_radius() const noexcept { return _shared_radius; }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] luisa::unique_ptr<Integrator::Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

class bxpm_inst final : public ProgressiveIntegrator::Instance {

public:
    using ProgressiveIntegrator::Instance::Instance;
    //A hash table for storing photons in grid
    //Some problem:can only initialize for super large photon cache(max_depth*photon_per_iter)
    //The fetchmax functions have wrong implementation in Luisa compute, so related feature are wrong now
    //(Including uint grid index, and inital_radius<0)
    class PhotonMap {
    private:
        Buffer<uint> _grid_head;
        Buffer<float> _beta;
        Buffer<float3> _wi;
        Buffer<float3> _position;
        Buffer<uint> _nxt;
        uint _size;       //size of maximum length
        Buffer<uint> _tot;//current photon count
        const Spectrum::Instance *_spectrum;
        Buffer<float> _grid_min;//atomic float3
        Buffer<float> _grid_max;//atomic float3
        Buffer<float> _grid_len;//the length of a single grid (float1)
        Buffer<float> _swl_lambda;
        Buffer<float> _swl_pdf;

        Buffer<float> _p_01;

    public:
        Buffer<uint> tot_test;
        PhotonMap(uint photon_count, const Spectrum::Instance *spectrum) {
            auto &&device = spectrum->pipeline().device();
            _grid_head = device.create_buffer<uint>(photon_count);
            _beta = device.create_buffer<float>(photon_count * spectrum->node()->dimension());
            _wi = device.create_buffer<float3>(photon_count);
            _position = device.create_buffer<float3>(photon_count);
            _nxt = device.create_buffer<uint>(photon_count);
            _tot = device.create_buffer<uint>(1u);
            _grid_len = device.create_buffer<float>(1u);
            _grid_min = device.create_buffer<float>(3u);
            _grid_max = device.create_buffer<float>(3u);
            _size = photon_count;
            _spectrum = spectrum;
            if (!_spectrum->node()->is_fixed()) {
                _swl_lambda = device.create_buffer<float>(photon_count * spectrum->node()->dimension());
                _swl_pdf = device.create_buffer<float>(photon_count * spectrum->node()->dimension());
            }
            tot_test = device.create_buffer<uint>(1u);
            _p_01 = device.create_buffer<float>(photon_count);
        }
        auto p_01(Expr<uint> index) const noexcept {
            return _p_01.read(index);
        }
        auto tot_photon() const noexcept {
            return _tot.read(0u);
        }
        auto grid_len() const noexcept {
            return _grid_len.read(0u);
        }
        auto size() const noexcept {
            return _size;
        }
        auto position(Expr<uint> index) const noexcept {
            return _position.read(index);
        }
        auto wi(Expr<uint> index) const noexcept {
            return _wi.read(index);
        }
        auto beta(Expr<uint> index) const noexcept {
            auto dimension = _spectrum->node()->dimension();
            SampledSpectrum s{dimension};
            for (auto i = 0u; i < dimension; ++i)
                s[i] = _beta.read(index * dimension + i);
            return s;
        }
        auto nxt(Expr<uint> index) const noexcept {
            return _nxt.read(index);
        }
        auto grid_head(Expr<uint> index) const noexcept {
            return _grid_head.read(index);
        }
        auto swl(Expr<uint> index) const noexcept {
            auto dimension = _spectrum->node()->dimension();
            SampledWavelengths swl(dimension);
            for (auto i = 0u; i < dimension; ++i) {
                swl.set_lambda(i, _swl_lambda.read(index * dimension + i));
                swl.set_pdf(i, _swl_pdf.read(index * dimension + i));
            }
            return swl;
        }
        void push(Expr<float3> position, SampledWavelengths swl, SampledSpectrum power, Expr<float3> wi, Float p_01) {
            $if(tot_photon() < size()) {
                auto index = _tot.atomic(0u).fetch_add(1u);
                auto dimension = _spectrum->node()->dimension();
                if (!_spectrum->node()->is_fixed()) {
                    for (auto i = 0u; i < dimension; ++i) {
                        _swl_lambda.write(index * dimension + i, swl.lambda(i));
                        _swl_pdf.write(index * dimension + i, swl.pdf(i));
                    }
                }
                _wi.write(index, wi);
                _position.write(index, position);
                for (auto i = 0u; i < dimension; ++i)
                    _beta.write(index * dimension + i, power[i]);
                for (auto i = 0u; i < 3u; ++i)
                    _grid_min.atomic(i).fetch_min(position[i]);
                for (auto i = 0u; i < 3u; ++i)
                    _grid_max.atomic(i).fetch_max(position[i]);
                _nxt.write(index, 0u);
                _p_01.write(index, p_01);
            };
        }
        //from uint3 grid id to hash index of the grid
        auto grid_to_index(Expr<int3> p) const noexcept {
            auto hash = ((p.x * 73856093) ^ (p.y * 19349663) ^
                         (p.z * 83492791)) %
                        (_size);
            return (hash + _size) % _size;
        }
        //from float3 position to uint3 grid id
        auto point_to_grid(Expr<float3> p) const noexcept {
            Float3 grid_min = {_grid_min.read(0),
                               _grid_min.read(1),
                               _grid_min.read(2)};
            return make_int3((p - grid_min) / grid_len()) + make_int3(2, 2, 2);
        }
        auto point_to_index(Expr<float3> p) const noexcept {
            return grid_to_index(point_to_grid(p));
        }
        void link(Expr<uint> index) {
            auto p = _position.read(index);
            auto grid_index = point_to_index(p);
            auto head = _grid_head.atomic(grid_index).exchange(index);
            _nxt.write(index, head);
        }
        void reset(Expr<uint> index) {
            _grid_head.write(index, ~0u);
            _tot.write(0, 0u);
            _nxt.write(index, ~0u);
            for (auto i = 0u; i < 3u; ++i) {
                _grid_min.write(i, std::numeric_limits<float>::max());
                _grid_max.write(i, -std::numeric_limits<float>::max());
            }
        }
        void write_grid_len(Expr<float> len) {
            _grid_len.write(0u, len);
        }
        auto split(Expr<float> grid_count) const noexcept {
            /* Float3 grid_min = {_grid_min.read(0),
                               _grid_min.read(1),
                               _grid_min.read(2)};
            Float3 grid_max = {_grid_max.read(0),
                               _grid_max.read(1),
                               _grid_max.read(2)};
            auto _grid_size = grid_max - grid_min;
            */
            auto _grid_size = _spectrum->pipeline().geometry()->world_max() - _spectrum->pipeline().geometry()->world_min();
            return min(min(_grid_size.x / grid_count, _grid_size.y / grid_count), _grid_size.z / grid_count);
        }
    };
    //Store the information of pixel updates
    class PixelIndirect {
        Buffer<float> _radius;
        Buffer<uint> _cur_n;
        Buffer<uint> _n_photon;
        Buffer<float> _phi;
        Buffer<float> _tau;
        const Film::Instance *_film;
        const Spectrum::Instance *_spectrum;
        bool _shared_radius;
        uint _photon_per_iter;
        float _clamp;

    public:
        PixelIndirect(uint photon_per_iter, const Spectrum::Instance *spectrum, const Film::Instance *film, float clamp, bool shared_radius) {
            _film = film;
            _spectrum = spectrum;
            _clamp = clamp;
            auto device = spectrum->pipeline().device();
            auto resolution = film->node()->resolution();
            auto dimension = 3u;//always save rgb
            _shared_radius = shared_radius;
            if (shared_radius) {
                _radius = device.create_buffer<float>(1);
                _cur_n = device.create_buffer<uint>(1);
                _n_photon = device.create_buffer<uint>(1);
            } else {
                _radius = device.create_buffer<float>(resolution.x * resolution.y);
                _cur_n = device.create_buffer<uint>(resolution.x * resolution.y);
                _n_photon = device.create_buffer<uint>(resolution.x * resolution.y);
            }
            _phi = device.create_buffer<float>(resolution.x * resolution.y * dimension);
            _tau = device.create_buffer<float>(resolution.x * resolution.y * dimension);
            _photon_per_iter = photon_per_iter;
        }
        void write_radius(Expr<uint2> pixel_id, Expr<float> value) noexcept {
            if (!_shared_radius) {
                auto resolution = _film->node()->resolution();
                _radius.write(pixel_id.y * resolution.x + pixel_id.x, value);
            } else {
                _radius.write(0u, value);
            }
        }
        void write_cur_n(Expr<uint2> pixel_id, Expr<uint> value) noexcept {
            if (!_shared_radius) {
                auto resolution = _film->node()->resolution();
                _cur_n.write(pixel_id.y * resolution.x + pixel_id.x, value);
            } else {
                _cur_n.write(0u, value);
            }
        }
        void write_n_photon(Expr<uint2> pixel_id, Expr<uint> value) noexcept {
            if (!_shared_radius) {
                auto resolution = _film->node()->resolution();
                _n_photon.write(pixel_id.y * resolution.x + pixel_id.x, value);
            } else {
                _n_photon.write(0u, value);
            }
        }
        void reset_phi(Expr<uint2> pixel_id) noexcept {
            auto resolution = _film->node()->resolution();
            auto offset = pixel_id.y * resolution.x + pixel_id.x;
            auto dimension = 3u;
            for (auto i = 0u; i < dimension; ++i)
                _phi.write(offset * dimension + i, 0.f);
        }
        void reset_tau(Expr<uint2> pixel_id) noexcept {
            auto resolution = _film->node()->resolution();
            auto offset = pixel_id.y * resolution.x + pixel_id.x;
            auto dimension = 3u;
            for (auto i = 0u; i < dimension; ++i)
                _tau.write(offset * dimension + i, 0.f);
        }
        auto radius(Expr<uint2> pixel_id) const noexcept {
            if (!_shared_radius) {
                auto resolution = _film->node()->resolution();
                return _radius.read(pixel_id.y * resolution.x + pixel_id.x);
            } else {
                return _radius.read(0u);
            }
        }
        //tau=(tau+clamp(phi))*value, see pixel_info_update for useage
        void update_tau(Expr<uint2> pixel_id, Expr<float> value) noexcept {
            auto resolution = _film->node()->resolution();
            auto offset = pixel_id.y * resolution.x + pixel_id.x;
            auto dimension = 3u;
            auto thershold = _clamp;
            for (auto i = 0u; i < dimension; ++i) {
                auto old_tau = _tau.read(offset * dimension + i);
                auto phi = _phi.read(offset * dimension + i);
                phi = max(-thershold, min(phi, thershold));//-thershold for wavelength sampling
                _tau.write(offset * dimension + i, (old_tau + phi) * value);
            }
        }

        auto n_photon(Expr<uint2> pixel_id) const noexcept {
            auto resolution = _film->node()->resolution();
            if (!_shared_radius) {
                return _n_photon.read(pixel_id.y * resolution.x + pixel_id.x);
            } else {
                return _n_photon.read(0u);
            }
        }
        auto cur_n(Expr<uint2> pixel_id) const noexcept {
            if (!_shared_radius) {
                auto resolution = _film->node()->resolution();
                return _cur_n.read(pixel_id.y * resolution.x + pixel_id.x);
            } else {
                return _cur_n.read(0u);
            }
        }
        auto phi(Expr<uint2> pixel_id) const noexcept {
            auto resolution = _film->node()->resolution();
            auto offset = pixel_id.y * resolution.x + pixel_id.x;
            auto dimension = 3u;
            Float3 ret;
            for (auto i = 0u; i < dimension; ++i)
                ret[i] = _phi.read(offset * dimension + i);
            return ret;
        }
        auto tau(Expr<uint2> pixel_id) const noexcept {
            auto resolution = _film->node()->resolution();
            auto offset = pixel_id.y * resolution.x + pixel_id.x;
            auto dimension = 3u;
            Float3 ret;
            for (auto i = 0u; i < dimension; ++i)
                ret[i] = _tau.read(offset * dimension + i);
            return ret;
        }
        void add_cur_n(Expr<uint2> pixel_id, Expr<uint> value) noexcept {
            if (!_shared_radius) {
                auto resolution = _film->node()->resolution();
                auto offset = pixel_id.y * resolution.x + pixel_id.x;
                _cur_n.atomic(offset).fetch_add(value);
            } else {
                _cur_n.atomic(0u).fetch_add(value);
            }
        }
        void add_phi(Expr<uint2> pixel_id, Expr<float3> phi) noexcept {
            auto resolution = _film->node()->resolution();
            auto offset = pixel_id.y * resolution.x + pixel_id.x;
            auto dimension = 3u;
            for (auto i = 0u; i < dimension; ++i)
                _phi.atomic(offset * dimension + i).fetch_add(phi[i]);
        }
        void pixel_info_update(Expr<uint2> pixel_id) {
            $if(cur_n(pixel_id) > 0) {
                Float gamma = 2.0f / 3.0f;
                // Float gamma = 1;
                UInt n_new = n_photon(pixel_id) + cur_n(pixel_id);
                Float r_new = radius(pixel_id) * sqrt(n_new * gamma / (n_photon(pixel_id) * gamma + cur_n(pixel_id)));
                //indirect.write_tau(pixel_id, (indirect.tau(pixel_id) + indirect.phi(pixel_id)) * (r_new * r_new) / (indirect.radius(pixel_id) * indirect.radius(pixel_id)));
                update_tau(pixel_id, r_new * r_new / (radius(pixel_id) * radius(pixel_id)));
                if (!_shared_radius) {
                    write_n_photon(pixel_id, n_new);
                    write_cur_n(pixel_id, 0u);
                    write_radius(pixel_id, r_new);
                }
                reset_phi(pixel_id);
            };
        }
        void shared_update() {
            auto pixel_id = make_uint2(0, 0);
            $if(cur_n(pixel_id) > 0) {
                Float gamma = 2.0f / 3.0f;
                // Float gamma = 1;
                UInt n_new = n_photon(pixel_id) + cur_n(pixel_id);
                Float r_new = radius(pixel_id) * sqrt(n_new * gamma / (n_photon(pixel_id) * gamma + cur_n(pixel_id)));
                write_n_photon(pixel_id, n_new);
                write_cur_n(pixel_id, 0u);
                write_radius(pixel_id, r_new);
            };
        }
    };

protected:
    void _render_one_camera(CommandBuffer &command_buffer, Camera::Instance *camera) noexcept override {
        if (!pipeline().has_lighting()) [[unlikely]] {
            LUISA_WARNING_WITH_LOCATION(
                "No lights in scene. Rendering aborted.");
            return;
        }
        auto spp = camera->node()->spp();
        auto resolution = camera->film()->node()->resolution();
        auto image_file = camera->node()->file();
        auto photon_per_iter = node<bxpm>()->photon_per_iter();
        auto pixel_count = resolution.x * resolution.y;
        auto spectrum = camera->pipeline().spectrum();
        //TODO: use sampler right
        uint add_x = (photon_per_iter + resolution.y - 1) / resolution.y;
        sampler()->reset(command_buffer, make_uint2(resolution.x + add_x, resolution.y), pixel_count + add_x * resolution.y, spp);
        command_buffer << pipeline().printer().reset();
        command_buffer << compute::synchronize();
        LUISA_INFO(
            "Rendering to '{}' of resolution {}x{} at {}spp.",
            image_file.string(),
            resolution.x, resolution.y, spp);

        using namespace luisa::compute;
        auto &&device = camera->pipeline().device();
        auto radius = node<bxpm>()->initial_radius();
        if (radius < 0) {
            auto _grid_size = spectrum->pipeline().geometry()->world_max() - spectrum->pipeline().geometry()->world_min();
            radius = min(min(_grid_size.x / -radius, _grid_size.y / -radius), _grid_size.z / -radius);
        }
        auto clamp = camera->film()->node()->clamp() * photon_per_iter * pi * radius * radius;
        PixelIndirect indirect(photon_per_iter, spectrum, camera->film(), clamp, node<bxpm>()->shared_radius());
        PhotonMap photons(photon_per_iter * node<bxpm>()->max_depth(), spectrum);
        a_pi = float(photon_per_iter * node<bxpm>()->rr_depth());
        float _outer_radius = radius;

        //initialize PixelIndirect
        Kernel2D indirect_initialize_kernel = [&]() noexcept {
            Buffer<float> _radius;
            Buffer<float> _cur_n;
            Buffer<float> _n_photon;
            Buffer<float> _phi;
            Buffer<float> _tau;
            auto index = dispatch_id().xy();
            auto radius = node<bxpm>()->initial_radius();
            if (radius < 0)
                photons.write_grid_len(photons.split(-radius));
            else
                photons.write_grid_len(node<bxpm>()->initial_radius());
            //camera->pipeline().printer().info("grid:{}", photons.grid_len());
            // indirect.write_radius(index, photons.grid_len());
            indirect.write_radius(index, _outer_radius); // TODO: bx hack
            //camera->pipeline().printer().info("rad:{}", indirect.radius(index));

            indirect.write_cur_n(index, 0u);
            indirect.write_n_photon(index, 0u);
            indirect.reset_phi(index);
            indirect.reset_tau(index);
        };
        //reset PhotonMap every spp
        Kernel1D photon_reset_kernel = [&]() noexcept {
            auto index = static_cast<UInt>(dispatch_x());
            photons.reset(index);
        };
        //put the photons into hash table
        Kernel1D photon_grid_kernel = [&]() noexcept {
            auto index = static_cast<UInt>(dispatch_x());
            auto radius = node<bxpm>()->initial_radius();
            $if(photons.nxt(index) == 0u) {
                photons.link(index);
            };
        };
        //emit photons
        Kernel2D photon_emit_kernel = [&](UInt frame_index, Float time) noexcept {
            auto pixel_id = dispatch_id().xy();
            auto sampler_id = UInt2(pixel_id.x + resolution.x, pixel_id.y);
            $if(pixel_id.x * resolution.y + pixel_id.y < photon_per_iter) {
                photon_tracing(photons, camera, frame_index, sampler_id, time);
            };
        };
        //check for direct and indirect(photon gathering)
        Kernel2D render_kernel = [&](UInt frame_index, Float time, Float shutter_weight) noexcept {
            //set_block_size(16u, 16u, 1u);
            auto pixel_id = dispatch_id().xy();
            auto L = Li(photons, indirect, camera, frame_index, pixel_id, time, shutter_weight);
            // camera->film()->accumulate(pixel_id, L, 0.5f);
        };
        //update the radius/light information per pixel
        Kernel2D indirect_update_kernel = [&]() noexcept {
            set_block_size(16u, 16u, 1u);
            auto pixel_id = dispatch_id().xy();
            indirect.pixel_info_update(pixel_id);
        };
        Kernel1D shared_update_kernel = [&]() noexcept {
            indirect.shared_update();
            photons.write_grid_len(indirect.radius(make_uint2(0, 0)));
        };
        //accumulate the stored indirect light into final image
        Kernel2D indirect_draw_kernel = [&](UInt tot_photon, UInt spp) noexcept {
            set_block_size(16u, 16u, 1u);
            auto pixel_id = dispatch_id().xy();
            auto L = get_indirect(indirect, camera->pipeline().spectrum(), pixel_id, tot_photon);
            // camera->film()->accumulate(pixel_id, L, 0.5f * spp);
            camera->film()->accumulate(pixel_id, L, 0.f);
        };
        bool enable_lt = node<bxpm>()->enable_lt;
        bool enable_rt = node<bxpm>()->enable_rt;
        bool enable_pm = node<bxpm>()->enable_pm;

        Kernel2D render_kernel_lt = [&](UInt frame_index, Float time, Float shutter_weight) noexcept {
            set_block_size(16u, 16u, 1u);
            auto pixel_id = dispatch_id().xy();
            if (enable_lt) {
                auto L = Li_vcm(camera, frame_index, pixel_id, time, shutter_weight, indirect);
            }
        };

        Kernel2D render_kernel_rt = [&](UInt frame_index, Float time, Float shutter_weight) noexcept {
            set_block_size(16u, 16u, 1u);
            auto pixel_id = dispatch_id().xy();
            if (enable_rt) {
                auto L = Li_vcm_rt(camera, frame_index, pixel_id, time, indirect);
                camera->film()->accumulate(pixel_id, L);
            }
            else {
                camera->film()->accumulate(pixel_id, make_float3());
            }
        };
        Clock clock_compile;
        auto render_lt = pipeline().device().compile(render_kernel_lt);
        auto render_rt = pipeline().device().compile(render_kernel_rt);
        auto render = pipeline().device().compile(render_kernel);
        auto update = pipeline().device().compile(indirect_update_kernel);
        auto shared_update = pipeline().device().compile(shared_update_kernel);
        auto indirect_draw = pipeline().device().compile(indirect_draw_kernel);
        auto indirect_initialize = pipeline().device().compile(indirect_initialize_kernel);
        auto indirect_update = pipeline().device().compile(indirect_update_kernel);
        auto photon_reset = pipeline().device().compile(photon_reset_kernel);
        auto photon_grid = pipeline().device().compile(photon_grid_kernel);
        auto emit = pipeline().device().compile(photon_emit_kernel);
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
        bool initial_flag = false;
        uint runtime_spp = 0u;
        //TODO: maybe swap the for order for better radius convergence
        for (auto s : shutter_samples) {
            pipeline().update(command_buffer, s.point.time);
            runtime_spp += s.spp;
            for (auto i = 0u; i < s.spp; i++) {
                command_buffer << render_lt(sample_id, s.point.time, s.point.weight).dispatch(resolution);
                command_buffer << render_rt(sample_id, s.point.time, s.point.weight).dispatch(resolution);
                //emit phtons then calculate L
                //TODO: accurate size reset
                command_buffer
                    << photon_reset().dispatch(photons.size());
                command_buffer << emit(sample_id, s.point.time)
                                    .dispatch(make_uint2(add_x, resolution.y));
                if (!initial_flag) {//wait for first world statistic
                    initial_flag = true;
                    command_buffer << indirect_initialize().dispatch(resolution);
                }
                command_buffer << photon_grid().dispatch(photons.size());
                command_buffer << render(sample_id, s.point.time, s.point.weight)
                                    .dispatch(resolution);
                command_buffer << update().dispatch(resolution);
                if (node<bxpm>()->shared_radius()) {
                    command_buffer << shared_update().dispatch(1u);
                }
                sample_id++;
                auto dispatches_per_commit =
                    display() && !display()->should_close() ?
                        node<ProgressiveIntegrator>()->display_interval() :
                        16;
                if (++dispatch_count % dispatches_per_commit == 0u) [[unlikely]] {
                    dispatch_count = 0u;
                    auto p = sample_id / static_cast<double>(spp);
                    if (display() && display()->update(command_buffer, sample_id)) {
                        progress.update(p);
                    } else {
                        command_buffer << [&progress, p] { progress.update(p); };
                    }
                }
            }
            command_buffer << pipeline().printer().retrieve();
        }
        LUISA_INFO("total spp:{}", runtime_spp);
        //tot_photon is photon_per_iter not photon_per_iter*spp because of unnormalized samples
        if (enable_pm) {
            command_buffer << indirect_draw(node<bxpm>()->photon_per_iter(), runtime_spp).dispatch(resolution);
        }
        command_buffer << synchronize();
        command_buffer << pipeline().printer().retrieve();

        progress.done();

        auto render_time = clock.toc();
        LUISA_INFO("Rendering finished in {} ms.", render_time);
    }

    [[nodiscard]] Float3 get_indirect(PixelIndirect &indirect, const Spectrum::Instance *spectrum, Expr<uint2> pixel_id, Expr<uint> tot_photon) noexcept {
        auto r = indirect.radius(pixel_id);
        auto tau = indirect.tau(pixel_id);
        Float3 L;
        L = tau / (tot_photon * pi * r * r);
        return L;
    }

    [[nodiscard]] Float3 Li(PhotonMap &photons, PixelIndirect &indirect, const Camera::Instance *camera, Expr<uint> frame_index,
                            Expr<uint2> pixel_id, Expr<float> time, Expr<float> shutter_weight) noexcept {
        sampler()->start(pixel_id, frame_index);
        auto u_filter = sampler()->generate_pixel_2d();
        auto u_lens = camera->node()->requires_lens_sampling() ? sampler()->generate_2d() : make_float2(.5f);
        auto [camera_ray, _, camera_weight] = camera->generate_ray(pixel_id, time, u_filter, u_lens);
        auto spectrum = pipeline().spectrum();
        auto swl = spectrum->sample(spectrum->node()->is_fixed() ? 0.f : sampler()->generate_1d());
        SampledSpectrum beta{swl.dimension(), shutter_weight * camera_weight};
        SampledSpectrum Li{swl.dimension()};
        SampledSpectrum testbeta{swl.dimension()};
        auto ray = camera_ray;
        auto pdf_bsdf = def(1e16f);
        auto hack_camera_ray = camera_ray;
        Float p_k;
        Float r_pm = indirect.radius(make_uint2());
        $for(depth, node<bxpm>()->max_depth()) {

            // trace
            auto wo = -ray->direction();
            auto it = pipeline().geometry()->intersect(ray);
            $if(!it->valid()) {
                $break;
            };

            $if(!it->shape().has_surface()) { $break; };

            // generate uniform samples
            auto u_light_selection = sampler()->generate_1d();
            auto u_light_surface = sampler()->generate_2d();
            auto u_lobe = sampler()->generate_1d();
            auto u_bsdf = sampler()->generate_2d();
            auto u_rr = def(0.f);
            auto rr_depth = node<bxpm>()->rr_depth();
            $if(depth + 1u >= rr_depth) { u_rr = sampler()->generate_1d(); };

            // sample one light
            auto light_sample = light_sampler()->sample(
                *it, u_light_selection, u_light_surface, swl, time);

            // trace shadow ray
            auto occluded = pipeline().geometry()->intersect_any(light_sample.shadow_ray);

            // evaluate material
            auto surface_tag = it->shape().surface_tag();
            auto eta_scale = def(1.f);
            Bool stop_direct = false;
            auto rr_threshold = node<bxpm>()->rr_threshold();
            auto q = max(beta.max() * eta_scale, .05f);
            $if(depth + 1u >= rr_depth) {
                $if(q < rr_threshold & u_rr >= q) { stop_direct = true; };
            };
            PolymorphicCall<Surface::Closure> call;
            pipeline().surfaces().dispatch(surface_tag, [&](auto surface) noexcept {
                surface->closure(call, *it, swl, wo, 1.f, time);
            });
            call.execute([&](auto closure) noexcept {
                // apply opacity map
                auto alpha_skip = def(false);
                if (auto o = closure->opacity()) {
                    auto opacity = saturate(*o);
                    alpha_skip = u_lobe >= opacity;
                    u_lobe = ite(alpha_skip, (u_lobe - opacity) / (1.f - opacity), u_lobe / opacity);
                }

                $if(alpha_skip) {
                    ray = it->spawn_ray(ray->direction());
                    pdf_bsdf = 1e16f;
                }
                $else {
                    if (auto dispersive = closure->is_dispersive()) {
                        $if(*dispersive) { swl.terminate_secondary(); };
                    }
                    //TODO: get this done
                    auto roughness = closure->roughness();
                    Bool stop_check;
                    if (node<bxpm>()->separate_direct()) {
                        stop_check = (roughness.x * roughness.y > 0.16f) | stop_direct;
                    } else {
                        stop_check = true;//always stop at first intersection
                    }
                    $if(depth > 0 & stop_check) {
                    // $if(stop_check) {
                        stop_direct = true;
                        auto grid = photons.point_to_grid(it->p());
                        $for(x, grid.x - 1, grid.x + 2) {
                            $for(y, grid.y - 1, grid.y + 2) {
                                $for(z, grid.z - 1, grid.z + 2) {
                                    Int3 check_grid{x, y, z};
                                    auto photon_index = photons.grid_head(photons.grid_to_index(check_grid));
                                    $while(photon_index != ~0u) {
                                        auto position = photons.position(photon_index);
                                        auto dis = distance(position, it->p());
                                        //pipeline().printer().info("check_grid:{},{},{};test_grid:{},{},{}; limit:{}", x, y, z, test_grid[0], test_grid[1], test_grid[2], indirect.radius(pixel_id));
                                        $if(dis <= indirect.radius(pixel_id)) {
                                            auto photon_wi = photons.wi(photon_index);
                                            auto photon_beta = photons.beta(photon_index);
                                            auto test_grid = photons.point_to_grid(position);
                                            auto eval_photon = closure->evaluate(wo, photon_wi);
                                            auto wi_local = it->shading().world_to_local(photon_wi);
                                            Float p_i = pi * sqr(r_pm) * inv_r(eval_photon.pdf) * a_pi;
                                            p_i = bx_power(p_i);
                                            Float w_heuristic = bx_heuristic(p_i, p_k + photons.p_01(photon_index));
                                            Float3 Phi;
                                            if (!spectrum->node()->is_fixed()) {
                                                auto photon_swl = photons.swl(photon_index);
                                                Phi = spectrum->wavelength_mul(swl, beta * (eval_photon.f / abs_cos_theta(wi_local)), photon_swl, photon_beta) * w_heuristic;
                                            } else {
                                                Phi = spectrum->srgb(swl, beta * photon_beta * eval_photon.f / abs_cos_theta(wi_local)) * w_heuristic;
                                            }
                                            //testbeta += Phi;
                                            indirect.add_phi(pixel_id, Phi);
                                            indirect.add_cur_n(pixel_id, 1u);
                                            //pipeline().printer().info("render:{}", indirect.cur_n(pixel_id));
                                        };

                                        photon_index = photons.nxt(photon_index);
                                    };
                                };
                            };
                        };
                    };
                    // sample material
                    auto surface_sample = closure->sample(wo, u_lobe, u_bsdf);
                    ray = it->spawn_ray(surface_sample.wi);
                    pdf_bsdf = surface_sample.eval.pdf;
                    auto w = ite(surface_sample.eval.pdf > 0.f, 1.f / surface_sample.eval.pdf, 0.f);
                    beta *= w * surface_sample.eval.f;
                    $if (depth == 0) {
                        auto pdf = closure->evaluate(surface_sample.wi, wo).pdf;
                        p_k = distance_squared(hack_camera_ray->origin(), it->p()) * ite(pdf > 0.f, 1.f / pdf, 0.f) * a_pk;
                        p_k = bx_power(p_k);
                    };
                    // apply eta scale
                    auto eta = closure->eta().value_or(1.f);
                    $switch(surface_sample.event) {
                        $case(Surface::event_enter) { eta_scale = sqr(eta); };
                        $case(Surface::event_exit) { eta_scale = sqr(1.f / eta); };
                    };
                };
            });
            beta = zero_if_any_nan(beta);
            $if(beta.all([](auto b) noexcept { return b <= 0.f; })) { $break; };
            $if(stop_direct) {
                $break;
            };
            $if(depth + 1u >= rr_depth) {
                beta *= ite(q < rr_threshold, 1.0f / q, 1.f);
            };
        };
        //return spectrum->srgb(swl, testbeta);//DEBUG
        return spectrum->srgb(swl, Li);
    }

    void photon_tracing(PhotonMap &photons, const Camera::Instance *camera, Expr<uint> frame_index,
                        Expr<uint2> pixel_id, Expr<float> time) {

        sampler()->start(pixel_id, frame_index);
        // generate uniform samples
        auto u_light_selection = sampler()->generate_1d();
        auto u_light_surface = sampler()->generate_2d();
        auto u_direction = sampler()->generate_2d();
        auto spectrum = pipeline().spectrum();
        auto swl = spectrum->sample(spectrum->node()->is_fixed() ? 0.f : sampler()->generate_1d());
        auto light_sample = light_sampler()->sample_le(
            u_light_selection, u_light_surface, u_direction, swl, time);
        //cos term canceled out in pdf
        SampledSpectrum beta = light_sample.eval.L / light_sample.eval.pdf;

        auto ray = light_sample.shadow_ray;
        auto pdf_bsdf = def(1e16f);
        Float p_0, p_1;
        $for(depth, node<bxpm>()->max_depth()) {

            // trace
            auto wi = -ray->direction();
            auto it = pipeline().geometry()->intersect(ray);
            $if (depth == 0) {
                auto light_sample = light_sampler()->sample(*it, u_light_selection, u_light_surface, swl, time);
                Float dist_sqr = compute::distance_squared(it->p(), ray->origin());
                // p_0 = inv_r(light_sample.eval.pdf); // p0hack
                p_0 = dist_sqr * inv_r(light_sample.eval.pdf) * a_p0;
                p_0 = bx_power(p_0);
            };

            // miss
            $if(!it->valid()) {
                $break;
            };

            $if(!it->shape().has_surface()) { $break; };

            // generate uniform samples
            auto u_lobe = sampler()->generate_1d();
            auto u_bsdf = sampler()->generate_2d();
            auto u_rr = def(0.f);
            auto rr_depth = node<bxpm>()->rr_depth();
            $if(depth + 1u >= rr_depth) { u_rr = sampler()->generate_1d(); };
            Bool condition = depth > 0;
            if (!node<bxpm>()->separate_direct()) {
                condition = depth >= 0;
            }
            $if (condition) {
                photons.push(it->p(), swl, beta, wi, p_0 + p_1);
            };
            // evaluate material
            auto surface_tag = it->shape().surface_tag();
            auto eta_scale = def(1.f);
            PolymorphicCall<Surface::Closure> call;
            pipeline().surfaces().dispatch(surface_tag, [&](auto surface) noexcept {
                surface->closure(call, *it, swl, wi, 1.f, time);
            });
            call.execute([&](auto closure) noexcept {
                // apply opacity map
                auto alpha_skip = def(false);
                if (auto o = closure->opacity()) {
                    auto opacity = saturate(*o);
                    alpha_skip = u_lobe >= opacity;
                    u_lobe = ite(alpha_skip, (u_lobe - opacity) / (1.f - opacity), u_lobe / opacity);
                }

                $if(alpha_skip) {
                    ray = it->spawn_ray(ray->direction());
                    pdf_bsdf = 1e16f;
                }
                $else {
                    if (auto dispersive = closure->is_dispersive()) {
                        $if(*dispersive) { swl.terminate_secondary(); };
                    }

                    // sample material
                    auto surface_sample = closure->sample(wi, u_lobe, u_bsdf, TransportMode::IMPORTANCE);
                    ray = it->spawn_ray(surface_sample.wi);
                    pdf_bsdf = surface_sample.eval.pdf;
                    auto w = ite(surface_sample.eval.pdf > 0.f, 1.f / surface_sample.eval.pdf, 0.f);
                    auto bnew = beta * w * surface_sample.eval.f;
                    // apply eta scale
                    auto eta = closure->eta().value_or(1.f);
                    $switch(surface_sample.event) {
                        $case(Surface::event_enter) { eta_scale = sqr(eta); };
                        $case(Surface::event_exit) { eta_scale = sqr(1.f / eta); };
                    };
                    eta_scale *= ite(beta.max() < bnew.max(), 1.f, bnew.max() / beta.max());
                    beta = bnew;
                    $if (depth == 0) {
                        auto dist = distance_squared(light_sample.shadow_ray->origin(), it->p());
                        auto pdf = closure->evaluate(surface_sample.wi, wi).pdf;
                        p_1 = dist * ite(pdf > 0.f, 1 / pdf, 0.f);
                        p_1 = bx_power(p_1);
                    };
                };
            });
            beta = zero_if_any_nan(beta);
            $if(beta.all([](auto b) noexcept { return b <= 0.f; })) { $break; };
            auto rr_threshold = node<bxpm>()->rr_threshold();
            auto q = max(eta_scale, .05f);
            $if(depth + 1u >= rr_depth) {
                $if(q < rr_threshold & u_rr >= q) { $break; };
                beta *= ite(q < rr_threshold, 1.0f / q, 1.f);
            };
        };
    }
    [[nodiscard]] Float3 Li_vcm(const Camera::Instance *camera, Expr<uint> frame_index,
                            Expr<uint2> pixel_id, Expr<float> time, Expr<float> shutter_weight, PixelIndirect &indirect) const noexcept {
        sampler()->start(pixel_id, frame_index);
        auto u_filter = sampler()->generate_pixel_2d();
        auto u_lens = camera->node()->requires_lens_sampling() ? sampler()->generate_2d() : make_float2(.5f);
        auto cs = camera->generate_ray(pixel_id, time, u_filter, u_lens);
        auto spectrum = pipeline().spectrum();
        auto swl = spectrum->sample(spectrum->node()->is_fixed() ? 0.f : sampler()->generate_1d());
        auto resolution = camera->film()->node()->resolution();

        auto u_light_selection = sampler()->generate_1d();
        // auto u_light_surface = sampler()->generate_2d();
        auto u_light_surface = make_float2(Float(pixel_id.x) / resolution.x, Float(pixel_id.y) / resolution.y);
        auto u_w = sampler()->generate_2d();
        Float cos_light;
        auto light_sample = light_sampler()->sample_ray(u_light_selection, u_light_surface, u_w, swl, time, cos_light);
        auto ray = light_sample.shadow_ray;
        auto beta = light_sample.eval.L / light_sample.eval.pdf;

        uint n_lt = node<bxpm>()->lt_depth;
        Float pd_l = inv_pi;
        // Float p_w = 1;
        Float p_1 = 1, p_0, p_i = 0;
        Float3 camera_normal = normalize(make_float3(0.5, -0.5, 1));
        // p_w /= pd_l;
        Float r_pm = indirect.radius(make_uint2());
        

        $for(depth, n_lt) {
            auto it = pipeline().geometry()->intersect(ray);
            $if (depth == 0) {
                auto light_sample = light_sampler()->sample(*it, u_light_selection, u_light_surface, swl, time);
                Float dist_sqr = compute::distance_squared(it->p(), ray->origin());
                // p_0 = inv_r(light_sample.eval.pdf); // p0hack
                p_0 = dist_sqr * inv_r(light_sample.eval.pdf) * a_p0; // p0hack
                p_0 = bx_power(p_0);
            };
            auto wi = -ray->direction();
            $if(!it->valid()) { $break; };
            $if(!it->shape().has_surface()) { $break; };

            $if (depth == 0) {
                auto dist = distance_squared(ray->origin(), it->p());
                // p_w /= (cos_light * abs_dot(it->ng(), wi) / dist); // G
            };
            // light tracing sample camera
            auto ray_connect = it->spawn_ray_to(cs.ray->origin());
            auto wo = ray_connect->direction();
            auto surface_tag = it->shape().surface_tag();
            auto occluded = pipeline().geometry()->intersect_any(ray_connect);
            $if (depth > 0) {
                PolymorphicCall<Surface::Closure> call;
                pipeline().surfaces().dispatch(surface_tag, [&](auto surface) noexcept {
                    surface->closure(call, *it, swl, wo, 1.f, time);
                });
                call.execute([&](const Surface::Closure *closure) noexcept {
                    auto eval = closure->evaluate(wi, wo);
                    $if(!occluded) {
                        Float importance;
                        auto pixel = camera->get_pixel(-ray_connect->direction(), time, make_float2(), Float2(), importance);
                        $if (pixel[0] < 32768) {
                            auto dist_sqr = distance_squared(it->p(), cs.ray->origin());
                            $if (depth >= node<bxpm>()->debug_depth) {
                                Float cos_eye = abs_dot(wo, camera_normal); // todo: bx2k hack
                                // Float p_w_tmp = 1.f;
                                // p_w_tmp *= (abs_dot(it->ng(), wo) * cos_eye / dist_sqr); // G
                                // p_w_tmp *= importance; // p_1
                                // $if (depth > 0) {
                                //     p_w_tmp *= closure->evaluate(wo, wi).pdf / abs_dot(it->ng(), wi);
                                // };
                                // Float p_i = 1.0f / (eval.pdf * importance / dist_sqr);
                                Float p_k = dist_sqr * ite(eval.pdf > 0, 1.0f / eval.pdf, 0.f) * a_pk;
                                p_k = bx_power(p_k);
                                Float w_heuristic = bx_heuristic(p_k, p_0 + p_1 + p_i);
                                camera->film()->accumulate(pixel, spectrum->srgb(swl, w_heuristic * beta * eval.f * importance / dist_sqr), 0.f);
                            };
                        };
                    };
                    auto roughness = closure->roughness();
                    $if (depth > 0 & roughness.x * roughness.y > 0.16f) {
                        auto eval = closure->evaluate(wo, wi);
                        p_i = pi * sqr(r_pm) * inv_r(eval.pdf) * a_pi;
                        p_i = bx_power(p_i);
                    };
                });
            };

            auto u_lobe = sampler()->generate_1d();
            auto u_bsdf = sampler()->generate_2d();
            // evaluate material
            auto eta_scale = def(1.f);
            PolymorphicCall<Surface::Closure> call;
            pipeline().surfaces().dispatch(surface_tag, [&](auto surface) noexcept {
                surface->closure(call, *it, swl, wi, 1.f, time);
            });
            call.execute([&](const Surface::Closure *closure) noexcept {
                // apply opacity map
                auto alpha_skip = def(false);
                if (auto o = closure->opacity()) {
                    auto opacity = saturate(*o);
                    alpha_skip = u_lobe >= opacity;
                    u_lobe = ite(alpha_skip, (u_lobe - opacity) / (1.f - opacity), u_lobe / opacity);
                }

                $if(alpha_skip) {
                    ray = it->spawn_ray(ray->direction());
                }
                $else {
                    if (auto dispersive = closure->is_dispersive()) {
                        $if(*dispersive) { swl.terminate_secondary(); };
                    }
                    // sample material
                    auto surface_sample = closure->sample(wi, u_lobe, u_bsdf);
                    ray = it->spawn_ray(surface_sample.wi);
                    auto w = ite(surface_sample.eval.pdf > 0.f, 1.f / surface_sample.eval.pdf, 0.f);
                    beta *= w * surface_sample.eval.f;
                    // apply eta scale
                    auto eta = closure->eta().value_or(1.f);
                    $switch(surface_sample.event) {
                        $case(Surface::event_enter) { eta_scale = sqr(eta); };
                        $case(Surface::event_exit) { eta_scale = sqr(1.f / eta); };
                    };
                    // p_w /= (surface_sample.eval.pdf / abs_dot(it->ng(), surface_sample.wi));
                    // $if (depth > 0) {
                    //     p_w *= closure->evaluate(surface_sample.wi, wi).pdf / abs_dot(it->ng(), wi);
                    // };
                    $if (depth == 0) {
                        auto dist = distance_squared(light_sample.shadow_ray->origin(), it->p());
                        // p_1 = dist / closure->evaluate(surface_sample.wi, wi).pdf / light_sample.eval.pdf;
                        // p_1 = dist / closure->evaluate(surface_sample.wi, wi).pdf;
                        // p_1 = dist;
                        auto pdf = closure->evaluate(surface_sample.wi, wi).pdf;
                        p_1 = dist * ite(pdf > 0.f, 1 / pdf, 0.f);
                        p_1 = bx_power(p_1);
                    };
                };
            });
            beta = zero_if_any_nan(beta);
            $if(beta.all([](auto b) noexcept { return b <= 0.f; })) { $break; };

            auto u_rr = def(0.f);
            auto rr_depth = node<bxpm>()->rr_depth();
            $if (depth + 1u >= rr_depth) { u_rr = sampler()->generate_1d(); };
            auto rr_threshold = node<bxpm>()->rr_threshold();
            auto q = max(beta.max() * eta_scale, .05f);
            $if(depth + 1u >= rr_depth) {
                $if(q < rr_threshold & u_rr >= q) { $break; };
                beta *= ite(q < rr_threshold, 1.0f / q, 1.f);
            };
        };
        return make_float3();
    }

    [[nodiscard]] Float3 Li_vcm_rt(const Camera::Instance *camera, Expr<uint> frame_index,
                            Expr<uint2> pixel_id, Expr<float> time, PixelIndirect &indirect) const noexcept {

        sampler()->start(pixel_id, frame_index);
        auto u_filter = sampler()->generate_pixel_2d();
        auto u_lens = camera->node()->requires_lens_sampling() ? sampler()->generate_2d() : make_float2(.5f);
        auto [camera_ray, _, camera_weight] = camera->generate_ray(pixel_id, time, u_filter, u_lens);
        auto spectrum = pipeline().spectrum();
        auto swl = spectrum->sample(spectrum->node()->is_fixed() ? 0.f : sampler()->generate_1d());
        SampledSpectrum beta{swl.dimension(), camera_weight};
        SampledSpectrum Li{swl.dimension()};
        SampledSpectrum Li_3{swl.dimension()};
        SampledSpectrum Li_4{swl.dimension()};

        Float3 camera_normal = normalize(make_float3(0.5, -0.5, 1));

        // Float p_w = 1;
        Float p_k = 0, p_k_next;
        Float r_pm = indirect.radius(make_uint2());
        Float camera_importance;
        Float cos_eye = abs_dot(camera_ray->direction(), camera_normal); // todo: bx2k hack
        {
            auto pixel = camera->get_pixel(camera_ray->direction(), time, make_float2(), Float2(), camera_importance);
            // p_w = 1 / importance;
        }

        auto ray = camera_ray;
        auto hack_camera_ray = camera_ray;
        auto pdf_bsdf = def(1e16f);
        Float3 ans_debug = make_float3();
        Float p_i = 0, p_i_next = -1;
        bool enable_i0 = node<bxpm>()->enable_i0;
        bool enable_i1 = node<bxpm>()->enable_i1;
        $for(depth, node<bxpm>()->max_depth()) {

            // trace
            auto wo = -ray->direction();
            auto it = pipeline().geometry()->intersect(ray);


            // miss
            $if(!it->valid()) {
                $break;
            };

            // hit light
            if (!pipeline().lights().empty() && enable_i0) {
                $if(it->shape().has_light()) {
                    auto eval = light_sampler()->evaluate_hit(*it, ray->origin(), swl, time);
                    Float dist_sqr = compute::distance_squared(it->p(), ray->origin());
                    // Float p_0 = dist_sqr * it->triangle_area();
                    Float p_0 = dist_sqr * inv_r(eval.pdf) * a_p0;
                    Float p_1 = dist_sqr * inv_r(pdf_bsdf);
                    p_0 = bx_power(p_0);
                    p_1 = bx_power(p_1);
                    Float w_heuristic = bx_heuristic(p_0, p_1 + p_k + p_i);
                    Li += beta * eval.L * w_heuristic;
                    // $if (depth == 5 & w_heuristic > .9f) {
                    //     // Li += p_0 / p_i * w_heuristic;
                    //     ans_debug.x = w_heuristic;
                    //     ans_debug.y = p_0;
                    //     // ans_debug.z = p_1 + p_i + p_k;
                    //     // ans_debug.z = p_i;
                    //     $if (p_i_next == 0) {
                    //         ans_debug.y = 128;
                    //     };
                    //     $if (p_i == 0) {
                    //         ans_debug.z = 128;
                    //     };
                    //     ans_debug = ans_debug;
                    // };
                };
            }

            $if (depth == 1) {
                p_k = p_k_next;
            };
            $if (p_i_next >= 0) {
                p_i = p_i_next;
            };

            $if(!it->shape().has_surface()) { $break; };

            // $if (depth == 0) {
                // p_w /= (cos_eye * abs_dot(it->ng(), ray->direction()) / distance_squared(ray->origin(), it->p())); // G
            // };

            // generate uniform samples
            auto u_light_selection = sampler()->generate_1d();
            auto u_light_surface = sampler()->generate_2d();
            auto u_lobe = sampler()->generate_1d();
            auto u_bsdf = sampler()->generate_2d();
            auto u_rr = def(0.f);
            auto rr_depth = node<bxpm>()->rr_depth();
            $if (depth + 1u >= rr_depth) { u_rr = sampler()->generate_1d(); };

            // sample one light
            auto light_sample = light_sampler()->sample(
                *it, u_light_selection, u_light_surface, swl, time);
            // Float dist_sqr = light_sample.shadow_ray->t_max();

            // trace shadow ray
            auto occluded = pipeline().geometry()->intersect_any(light_sample.shadow_ray);

            // evaluate material
            auto surface_tag = it->shape().surface_tag();
            auto eta_scale = def(1.f);
            PolymorphicCall<Surface::Closure> call;
            pipeline().surfaces().dispatch(surface_tag, [&](auto surface) noexcept {
                surface->closure(call, *it, swl, wo, 1.f, time);
            });
            call.execute([&](const Surface::Closure *closure) noexcept {

                // apply opacity map
                auto alpha_skip = def(false);
                if (auto o = closure->opacity()) {
                    auto opacity = saturate(*o);
                    alpha_skip = u_lobe >= opacity;
                    u_lobe = ite(alpha_skip, (u_lobe - opacity) / (1.f - opacity), u_lobe / opacity);
                }

                $if(alpha_skip) {
                    ray = it->spawn_ray(ray->direction());
                    pdf_bsdf = 1e16f;
                }
                $else {
                    if (auto dispersive = closure->is_dispersive()) {
                        $if(*dispersive) { swl.terminate_secondary(); };
                    }
                    // direct lighting
                    auto wi = light_sample.shadow_ray->direction();
                    auto eval = closure->evaluate(wo, wi);
                    if (enable_i1) {
                        $if(light_sample.eval.pdf > 0.0f & !occluded) {
                            Float cos_light = abs(wi.y); // hack
                            Float pd_l = inv_pi;
                            auto it_tmp = pipeline().geometry()->intersect(it->spawn_ray(wi));
                            Float dist_sqr = compute::distance_squared(it->p(), it_tmp->p());
                            // Float p_w_tmp = 1.f;
                            // p_w_tmp *= (abs_dot(it->ng(), wi) * cos_light / dist_sqr); // G
                            // p_w_tmp *= pd_l;
                            // $if (depth > 0) {
                            //     p_w_tmp *= closure->evaluate(wi, wo).pdf / abs_dot(it->ng(), wo);
                            // };
                            // Float p_i = 1.0f / (eval.pdf * pd_l / dist_sqr);
                            Float p_0 = dist_sqr * inv_r(light_sample.eval.pdf) * a_p0;
                            Float p_1 = dist_sqr * inv_r(eval.pdf);
                            p_0 = bx_power(p_0);
                            p_1 = bx_power(p_1);
                            // Float p_1 = 1;
                            // Float sqr_heuristic = 1 / (1 + sqr(p_w * p_w_tmp));
                            Float w_heuristic = bx_heuristic(p_1, p_0 + p_k + p_i);
                            // bool enable_lt = node<bxpm>()->enable_lt;
                            // if (!enable_lt) sqr_heuristic = 1;
                            auto w = inv_r(light_sample.eval.pdf);
                            Li += w_heuristic * w * beta * eval.f * light_sample.eval.L;
                            // $if (p_1 < 0) {
                            //     ans_debug.x += 1;
                            // };
                            // $if (p_k < 0) {
                            //     ans_debug.y += 1;
                            // };
                            // $if (delta < 0) {
                            //     ans_debug.z += 1;
                            // };
                            // for (int k = 0; k < 3; k++) {
                            //     $if (delta[k] < 0) {
                            //         ans_debug[k] += 1;
                            //     };
                            // }
                            // $if (depth == 3) {
                            //     Li_3 = w * beta * eval.f * light_sample.eval.L;
                            // };
                            // $if (depth == 4) {
                            //     Li_4 = sqr_heuristic * w * beta * eval.f * light_sample.eval.L;
                            // };
                        };
                    }
                    // sample material
                    auto surface_sample = closure->sample(wo, u_lobe, u_bsdf);
                    ray = it->spawn_ray(surface_sample.wi);
                    pdf_bsdf = surface_sample.eval.pdf;
                    auto w = ite(surface_sample.eval.pdf > 0.f, 1.f / surface_sample.eval.pdf, 0.f);
                    beta *= w * surface_sample.eval.f;
                    // p_w /= (pdf_bsdf / abs_dot(surface_sample.wi, it->ng()));
                    // $if (depth > 0) {
                    //     p_w *= closure->evaluate(surface_sample.wi, wo).pdf / abs_dot(it->ng(), wo);
                    // };
                    $if (depth == 0) {
                        // p_k = distance_squared(hack_camera_ray->origin(), it->p()) / closure->evaluate(surface_sample.wi, wo).pdf / camera_importance;
                        // auto pdf = closure->evaluate(surface_sample.wi, wo).pdf;
                        auto pdf = surface_sample.eval.pdf;
                        p_k_next = distance_squared(hack_camera_ray->origin(), it->p()) * ite(pdf > 0.f, 1.f / pdf, 0.f) * a_pk;
                        p_k_next = bx_power(p_k_next);
                    };
                    auto roughness = closure->roughness();
                    $if (depth > 0 & p_i_next < 0 & roughness.x * roughness.y > 0.16f) {
                        p_i_next = pi * sqr(r_pm) * inv_r(pdf_bsdf) * a_pi;
                        p_i_next = bx_power(p_i_next);
                    };
                    // apply eta scale
                    auto eta = closure->eta().value_or(1.f);
                    $switch(surface_sample.event) {
                        $case(Surface::event_enter) { eta_scale = sqr(eta); };
                        $case(Surface::event_exit) { eta_scale = sqr(1.f / eta); };
                    };
                };
            });
            beta = zero_if_any_nan(beta);
            $if(beta.all([](auto b) noexcept { return b <= 0.f; })) { $break; };
            auto rr_threshold = node<bxpm>()->rr_threshold();
            auto q = max(beta.max() * eta_scale, .05f);
            $if(depth + 1u >= rr_depth) {
                $if(q < rr_threshold & u_rr >= q) { $break; };
                beta *= ite(q < rr_threshold, 1.0f / q, 1.f);
            };
        };
        // return spectrum->srgb(swl, Li);
        return spectrum->srgb(swl, Li) + ans_debug;
        // Float3 ans_3 = spectrum->srgb(swl, Li_3);
        // Float3 ans_4 = spectrum->srgb(swl, Li_4);
        // Float3 ans = spectrum->srgb(swl, Li);
        // Float tmp = ans_3.x;
        // $if (isnan(tmp)) {
        //     tmp = 0;
        // };
        // return make_float3(tmp, 0.f, ans.z);
        // return make_float3(0.f, 0.f, ans.z);
    }
};

luisa::unique_ptr<Integrator::Instance> bxpm::build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    return luisa::make_unique<bxpm_inst>(
        pipeline, command_buffer, this);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::bxpm)
