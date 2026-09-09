"""Stub: storage operations not needed for evaluation pipeline."""


class _StorageStub:
    """No-op storage client for eval-only deployment."""

    def upload(self, *args, **kwargs):
        raise NotImplementedError("Storage not available in eval-only deployment")

    def download(self, *args, **kwargs):
        raise NotImplementedError("Storage not available in eval-only deployment")

    def get_signed_url(self, *args, **kwargs):
        raise NotImplementedError("Storage not available in eval-only deployment")


storage_client = _StorageStub()
