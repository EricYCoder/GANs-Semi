from tqdm import tqdm


class Progress(tqdm):
    def set_update_hook(self, hook):
        self.hook = hook

    def update(self, n=1):
        super().update(n)
        try:
            self.hook(self.n, self.total)
        except AttributeError:
            pass

    def update_with_total(self, n, total):
        if total is not None:
            self.total = total
        self.update(n)

    def update_to(self, n, total=None):
        self.update_with_total(n - self.n, total)


class ProgressStream:
    def write(self, s):
        Progress.write(s, end="")
        pass

    def flush(self):
        pass
