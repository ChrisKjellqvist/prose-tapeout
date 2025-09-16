#include <coroutine>

struct promise;

struct coroutine : std::coroutine_handle<promise> {
  using promise_type = ::promise;
};

struct promise {
  bool value_;
  coroutine get_return_object() { return {coroutine::from_promise(*this)}; }
  std::suspend_always initial_suspend() noexcept { return {}; }
  std::suspend_always final_suspend() noexcept { return {}; }
  void unhandled_exception() {}
};

struct S
{
    coroutine f(int i = 0)
    {
        if (i++ < 10) {
          co_yield true;
        }
        co_return;
    }
};

