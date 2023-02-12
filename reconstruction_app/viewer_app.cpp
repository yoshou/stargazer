#include <functional>
#include "viewer.hpp"

const int SCREEN_WIDTH = 1680;
const int SCREEN_HEIGHT = 1050;

struct reconstruction_viewer : public window_base
{
    std::mutex mtx;

    reconstruction_viewer()
        : window_base("Reconstruction Viewer", SCREEN_WIDTH, SCREEN_HEIGHT)
    {
    }

    virtual void initialize() override
    {
        window_base::initialize();
    }

    virtual void on_close() override
    {
        std::lock_guard<std::mutex> lock(mtx);
        window_manager::get_instance()->exit();
        window_base::on_close();
    }

    virtual void on_scroll(double x, double y) override
    {
    }

    virtual void update() override
    {
        std::lock_guard<std::mutex> lock(mtx);
        if (handle == nullptr)
        {
            return;
        }
    }

    virtual void on_char(unsigned int codepoint) override
    {
    }
};

static std::vector<std::function<void()>> on_shutdown_handlers;
static std::atomic_bool exit_flag(false);

static void shutdown()
{
    std::for_each(std::rbegin(on_shutdown_handlers), std::rend(on_shutdown_handlers), [](auto handler)
                  { handler(); });
    exit_flag.store(true);
}

int reconstruction_viewer_main()
{
    const auto win_mgr = window_manager::get_instance();
    win_mgr->initialize();

    on_shutdown_handlers.push_back([win_mgr]()
                                   { win_mgr->terminate(); });

    const auto viewer = std::make_shared<reconstruction_viewer>();

    const auto rendering_th = std::make_shared<rendering_thread>();
    rendering_th->start(viewer.get());

    on_shutdown_handlers.push_back([rendering_th, viewer]()
                                   {
        rendering_th->stop();
        viewer->destroy(); });

    while (!win_mgr->should_close())
    {
        win_mgr->handle_event();
    }

    shutdown();

    return 0;
}

int main()
{
    return reconstruction_viewer_main();
}
