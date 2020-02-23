import unittest

import OAA_Utils

class OAA_Util_test(unittest.TestCase):

    def test_make_accumulated_path(self):
        image_paths = ["C://image//image.jpg", "C://image//image.jpg"]
        image_dir = "C://image"
        OAA_dir = "C://OAA"
        result = ["C://OAA//image.jpg", "C://OAA//image.jpg"]
        
        OAA_paths = OAA_Utils.make_accumulated_image_path(image_paths, image_dir, OAA_dir)
        self.assertEqual(OAA_paths, result)
    
if __name__ == "__main__":
    unittest.main()