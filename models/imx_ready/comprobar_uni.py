import pkg_resources

print("uni version:", pkg_resources.get_distribution("uni").version)
from uni.common import logger
print("setup_uni_logging" in dir(logger))
