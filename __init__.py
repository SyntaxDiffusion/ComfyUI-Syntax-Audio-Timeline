from comfy_api.latest import ComfyExtension
from typing_extensions import override

from .nodes import NODES

WEB_DIRECTORY = "./web"


class SyntaxActionCurves(ComfyExtension):
    @override
    async def get_node_list(self):
        return NODES


async def comfy_entrypoint():
    return SyntaxActionCurves()
