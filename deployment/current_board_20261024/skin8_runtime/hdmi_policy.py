"""Enable original HDMI lifecycle; HTTP handlers serialize hardware transitions."""

def guard_camus(web):
    # camus.web.Handler.do_POST already owns server.busy for the whole HDMI
    # operation. Reacquiring its non-reentrant semaphore here rejects all shows.
    original = web.CamusServer.start_hdmi
    def start(self, *args, **kwargs):
        if self.fault:
            raise ValueError('硬件状态异常，请检查服务')
        return original(self, *args, **kwargs)
    web.CamusServer.start_hdmi = start

def guard_skin(web):
    # Original Console serializes active work and releases SkinWorker first.
    pass
