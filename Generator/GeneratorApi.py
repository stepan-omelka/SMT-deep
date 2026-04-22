from typing import Tuple

from Generator.Utils.KernToPng import KernToPng
from Generator.Utils.KernGenerator import KernGenerator
from Generator.Utils.KernAnnotationParser import convert_kern_to_annotated

class Generator:
    def __init__(self):
        self.kg = KernGenerator()
        self.kern_2_png = KernToPng()

    def generateStaff(self) -> Tuple[str, str]:
        kern: str = str(self.kg.generate(num_measures=4))
        image_path: str = self.kern_2_png.render_to_image(kern)
        annotated_kern: str = convert_kern_to_annotated(kern)
        return image_path, annotated_kern


if __name__ == "__main__":
    gen = Generator()
    image_path, kern = gen.generateStaff()
    print(kern)