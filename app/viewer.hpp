#pragma once

#include <memory>
#include <thread>
#include <mutex>
#include <atomic>
#include <queue>

struct window_base;

struct graphics_context
{
    window_base *window;

    graphics_context(window_base *window)
        : window(window)
    {
    }

    virtual void attach();
    virtual void detach();
    virtual void clear();
    virtual void swap_buffer();
    virtual ~graphics_context();
};

struct window_base
{
public:
    void *handle;
    std::string name;
    std::size_t width;
    std::size_t height;

    std::mutex mtx;

    window_base(std::string name, std::size_t width, std::size_t height);

    void *get_handle() const;

    virtual bool is_closed() const;

    virtual void on_close();
    virtual void on_key(int key, int scancode, int action, int mods);
    virtual void on_char(unsigned int codepoint);
    virtual void on_scroll(double x, double y);
    virtual void on_mouse_click(int button, int action, int mods);
    virtual void on_mouse(double x, double y);
    virtual void on_enter(int entered);
    virtual void on_resize(int width, int height);
    virtual void show();
    virtual void initialize();
    virtual void destroy();
    virtual void update();
    virtual ~window_base() = default;

    graphics_context create_graphics_context();
};

struct window_manager
{
    std::thread::id thread_id;
    std::atomic_bool should_close_flag;

    struct action_func_base
    {
        virtual void invoke(){};
        virtual ~action_func_base() = default;
    };

    template <typename Func>
    struct action_func : public action_func_base
    {
        Func func;
        action_func(Func &&_func) : func(std::move(_func)) {}
        action_func(action_func &&other) : func(std::move(other.func)) {}
        action_func &operator=(action_func &&other)
        {
            func = std::move(other.func);
            return *this;
        }
        virtual void invoke()
        {
            func();
        };
        virtual ~action_func() = default;
    };

    std::deque<std::unique_ptr<action_func_base>> queue;

    window_manager();

    bool should_close();
    virtual void handle_event();
    void *create_window_handle(std::string name, int width, int height, window_base *window);
    void destroy_window_handle(void *handle);
    void show_window(void *handle);
    void hide_window(void *handle);
    virtual void initialize();
    virtual void exit();
    virtual void terminate();

    virtual ~window_manager() = default;

    static std::shared_ptr<window_manager> get_instance();
};

struct rendering_thread
{
    std::unique_ptr<std::thread> th;

public:
    rendering_thread()
    {
    }

    void start(window_base *window)
    {
        th = std::make_unique<std::thread>([window]()
                                           {
            window->initialize();
            auto graphics_ctx = window->create_graphics_context();
            graphics_ctx.attach();
            window->show();
            while (!window->is_closed())
            {
                graphics_ctx.clear();
                window->update();
                graphics_ctx.swap_buffer();
            } });
    }

    void stop()
    {
        if (th && th->joinable())
        {
            th->join();
        }
    }
};

struct mouse_state
{
    double x, y;
    int right_button;
    int middle_button;
    int left_button;

    static mouse_state get_mouse_state(void *handle);
};
