"""Signal handlers request shutdown; the inference thread finishes before DMA release."""
import signal
import threading


def graceful_signals(server):
    if threading.current_thread() is not threading.main_thread():return
    def stop(signum,frame):
        threading.Thread(target=server.shutdown,name='acdc-stop',daemon=True).start()
    signal.signal(signal.SIGTERM,stop)
