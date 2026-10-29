"""یک‌بار اجرا شود: مدل MiniLM را در پوشه‌ی models دانلود می‌کند (حدود ۱۲۰ مگابایت)."""
import core


def show(done, total):
    if total:
        print(f"\r  {done / 1e6:.0f} / {total / 1e6:.0f} MB", end="", flush=True)


if __name__ == "__main__":
    core.download_model("minilm", progress=show)
    print("\nDone.")
