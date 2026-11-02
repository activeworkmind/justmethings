"""Background saver for the review app: writes the final photo as soon as a slot is approved (or changed / rejected),
so 'Approve + Next' stays instant while the file lands in final_output/ a moment later.

One worker thread, one queue. A slot queued twice before it is picked up is rendered once (with its latest
decision); a slot edited while it renders is queued again. Results come back through `done` and are read by the
GUI's main thread (Tk must not be touched from here)."""
import queue
import threading

from . import finalize


class Saver:
    def __init__(self):
        self.done = queue.Queue()                  # (key, result dict | None, error str | None)
        self._q = queue.Queue()
        self._pending = set()
        self._lock = threading.Lock()
        self._active = 0
        threading.Thread(target=self._run, daemon=True).start()

    def submit(self, key: str) -> None:
        with self._lock:
            if key in self._pending:               # already queued: the worker will read the latest decision
                return
            self._pending.add(key)
        self._q.put(key)

    @property
    def busy(self) -> int:
        """Slots queued or being rendered right now."""
        with self._lock:
            return len(self._pending) + self._active

    def _run(self):
        while True:
            key = self._q.get()
            with self._lock:
                self._pending.discard(key)         # before rendering, so an edit during the render re-queues the slot
                self._active += 1
            try:
                self.done.put((key, finalize.export_slot(key), None))
            except Exception as e:
                self.done.put((key, None, f"{type(e).__name__}: {e}"))
            finally:
                with self._lock:
                    self._active -= 1
