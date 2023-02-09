import numpy as np
import random
import copy
import matplotlib.pyplot as plt
'''

Problem:
    Un-callibrated Galvo lenses increases the error of guiding the laser light from source to target, especially on the edges of the lense.
    For example, when controlling an un-callibrated galvo lense to mark a "Square" shape on the target, the error causes the shape to distort
    into other unwanted shapes. The further the Galvo guide the light away from the center, the larger margin of error. 
Purpose:
    To implement Galvo Calibration algorithm with the help of CCD camera which provides the absolute error of controlled points 
    marked from the laser source

Pre-requisit:
    1. Galvo Lenses: - Controls the laser light from source to target via analogue signal
                     - Analogue signal is a 2D (x,y)coordinate system from the range of 0 to 2^16 (65536)
                     - **Required** - analogue signal to mm conversion (maybe dependant on focus lense)
    2. Focus lense:  - Lense used to focus laser light, impacted via distance from the lense to the target
                     - True Field:  - 校正檔有效範圍，對應的實際長度，也就是 MarkingMate 校 正過程中，所輸入的「最短中心線長度」。
                                    - i.e. 100 = 100 mm x 100 mm
                     - K:           - 校正檔使用範圍與全區域的比值。
    4. CCD camera:   - Located on the side of the Galvo Lense perpendicular to the surface of the target
                     - Provides image of the target in pixels
                     - **Required** - pixel to mm conversion
    5. XY Table:     - High precise table that can move the target underneath the galvo lense (where laser marking is applied)
                     - **Required** - Alignment between CCD center and Galvo center (i.e. when marking a point via galvo lense, 
                                      the CCD image center should be right on the point)
    6. Point Capture:- Software used to capture the points using the CCD camera's image
       Software
How:
    1. Create a field of laser marked points i.e. 9x9=81 (prefer odd number) evenly distributed on the target via controlling the galvo lenses
        - Note: the absolute ideal location of all 81 points is known from the analogue signal to mm conversion.
        - Following is analogue points - **Required** - convert them to mm from the center
        - Note 1: CENTER POINT - (32764 analogue,32764 analogue) -> (0 mm , 0 mm)
        - Note 2: TOP LEFT POINT - (0 analogue,0 analogue) -> (-50 mm , 50 mm) assuming true field = 100 mm
         |------------------------------------------------------------------------------|
         |(0,0)      |(8191,0)      |(16382,0) ....     |(57337,0)      |(65535, 0)     |
         |(0, 8191)  |(8191, 8191)  |(16382,8191) ....  |(57337,8191)   |(65535, 8191)  |
         |(0, 16382) |(8191, 16382) |(16382,16382) .... |(57337,16382)  |(65535, 16382) |
         |.          |              |                   |               |               |
         |.          |              |                   |               |               |         
         |.          |              |                   |               |               |
         |(0, 57337) |              |                   |               |               |
         |(0, 65535) (8191, 65535) (16382, 65535) .... (57337,65535)    |(65535, 65535) |
         -------------------------------------------------------------------------------|
    2. Using the CCD camera, capture the image with the CENTER POINT (32764 analogue,32764 analogue) as the center of the CCD image (in pixels)
    3. Identify the un-callibrated absolute coordinates (mm) of each point marked by the Galvo in step 1
        - requires point capture software of converting points to pixel coordinates i.e.(1024px, 1024px)
        - convert these pixel coordinates into absolute coordinates (mm)
    4. Calculate the ideal absolute distance (mm) from the center of each point of where the points should actually be
    5. Obtain the absolute difference (mm) between the ideal and un-callibrated coordinates of the points marked by the galvo
    6. Convert back into analogue unit and feed it back as a COR file for all points (9x9=81)

Out of Scope:
    1. 校正查表演算法 格點法 - Not required
    2. ZComp: Z 補償偏位

Output: COR 檔案資訊
    XComp: X 補償偏位
    YComp: Y 補償偏位
    

'''
class Visualise():
    def plot_scatter(self, multi_coordinates: list):
        labels = ['true_coord', 'filtered_true_coord', 'lens_coord', 'ideal_coord', 'mock_coord']
        markers= ['.', '*', 'x', '+', 'P']
        index = 0
        if len(multi_coordinates) > len(labels):
            raise IndexError("too many coordinates, please increase labels and markers in function")
        for coordinates in multi_coordinates:
            plt.scatter(coordinates[0], coordinates[1], marker=markers[index], label=labels[index])
            index = index + 1
        plt.show()

class Coordinate():
    def __init__(self, true_coord = True, true_field: int = 100, res: int = 8, K: int = 0.5):
        self.max_pixel = 2**16
        self.res = res
        self.true_field = true_field
        self.K = K
        if true_coord:
            max_field = self.true_field / self.K
            self.Y_mm, self.X_mm  = self.create_coords_mm(length = max_field, step = max_field / (self.res ** 2), start = 0)
            self.X_true_coord_pixel = self.coord_mm_to_pixel_conversion(self.X_mm)
            self.Y_true_coord_pixel = self.coord_mm_to_pixel_conversion(self.Y_mm)
        else:
            self.Y_mm, self.X_mm = np.empty
            self.X_true_coord_pixel, self.Y_true_coord_pixel = np.empty

    def create_coords_mm(self, length: int, step: float, start: int) -> np.array:
        '''
        Returns X and Y in 2D array
        '''
        steps = np.arange(start=start, stop=start + length + step, step=step)
        Y,X = np.meshgrid(steps,steps)
        return Y,X

    # def get_coord_mm(self, x_step: int, y_step: int) -> tuple:
    #     return (self.X[x_step], self.Y[y_step])

    def mm_to_pixel_conversion(self, mm_val: int) -> tuple:
        '''
        圖形座標 mm 轉換成振鏡位置 Pixel 公式
        Note: Range from -half of lens collaboration length to +half of lens collaboration length, starting at the centre
        '''
        pixel_val = mm_val * self.max_pixel/self.true_field * self.K + self.max_pixel / 2
        return pixel_val

    def coord_mm_to_pixel_conversion(self, coord_mm: np.array) -> tuple:
        '''
        圖形座標 mm 轉換成振鏡位置 Pixel 公式
        '''
        local_coord_mm = copy.deepcopy(coord_mm)
        max_field = self.true_field / self.K
        local_coord_mm = coord_mm * (self.max_pixel/max_field)
        return local_coord_mm.astype(int)

    def filtered_true_coord_mm(self):
        half_true_field = self.true_field / 2
        X_true_filtered = self.X_mm[(self.X_mm >= self.true_field-half_true_field) & (self.X_mm <= self.true_field+half_true_field)]
        Y_true_filtered = self.Y_mm[(self.Y_mm >= self.true_field-half_true_field) & (self.Y_mm <= self.true_field+half_true_field)]
        return X_true_filtered, Y_true_filtered

    def add_error_to_coord_mm(self, X_2d_arr, Y_2d_arr, max_error_val : float):
        X_error = copy.deepcopy(X_2d_arr)
        Y_error = copy.deepcopy(Y_2d_arr)
        for idx, _ in np.ndenumerate(X_error):
            X_error[idx] = X_error[idx] + max_error_val * random.uniform(-1,1)
            Y_error[idx] = Y_error[idx] + max_error_val * random.uniform(-1,1)
        return X_error, Y_error

    def callibration(self, X_pixel, Y_pixel):
        pixel_resolution = (self.max_pixel / self.res**2)
        start_id = 16
        # i and j identify the bottem left coordinates on the true coordination in pixels which will be used in calculation to identify X' and Y'
        i = (X_pixel / pixel_resolution).astype(int)
        j = (Y_pixel / pixel_resolution).astype(int)

        # Note: 0x3ff is 1023 - and is used to identify how much offset the point is to the bottom left coordinate on X and Y axis
        delta = 0x3ff
        dX = X_pixel & delta
        dY = Y_pixel & delta
        
        for idx, _ in np.ndenumerate(i):
            Xca = self.X_true_coord_pixel[(idx[0] + start_id, idx[1] + start_id + 1)]
            Xcb = self.X_true_coord_pixel[(idx[0] + start_id + 1, idx[1] + start_id + 1)]
            Xcc = self.X_true_coord_pixel[(start_id, start_id)]
            Xcd = self.X_true_coord_pixel[(idx[0] + start_id + 1, idx[1] + start_id)]

            Xcab = ((Xcb - Xca) * dX[idx]) /1024 + Xca
            Xccd = ((Xcd - Xcc) * dX[idx]) /1024 + Xcc
            X_comp = X_pixel[idx] + ((Xcab - Xccd) * dY[idx])/1024 + Xccd
        
        return X_comp

if __name__ == "__main__":
    true_field = 100
    res = 8
    K = 0.5
    step_lens = true_field / res
    start_lens = true_field/2
    max_error_val = step_lens / 2

    true_coord = Coordinate(true_field = true_field, res = res, K = K, true_coord=True)
    X_filtered_true_coord_mm, Y_filtered_true_coord_mm = true_coord.filtered_true_coord_mm()
    X_lens, Y_lens = true_coord.create_coords_mm(step = step_lens, length=true_field, start=start_lens)
    X_ideal, Y_ideal = true_coord.create_coords_mm(step = step_lens, length=true_field - step_lens, start=start_lens + step_lens / 2)
    X_mock, Y_mock = true_coord.add_error_to_coord_mm(X_ideal,Y_ideal, max_error_val = max_error_val)

    visualise = Visualise()
    # visualise.plot_scatter([
    #     [true_coord.X_mm, true_coord.Y_mm],
    #     [X_filtered_true_coord_mm, Y_filtered_true_coord_mm],
    #     [X_lens, Y_lens],
    #     [X_ideal, Y_ideal],
    #     [X_mock, Y_mock]
    # ])

    # Convert mm unit into pixels
    X_filtered_true_coord_pixel = true_coord.coord_mm_to_pixel_conversion(X_filtered_true_coord_mm)
    Y_filtered_true_coord_pixel = true_coord.coord_mm_to_pixel_conversion(Y_filtered_true_coord_mm)
    X_lens_pixel = true_coord.coord_mm_to_pixel_conversion(X_lens)
    Y_lens_pixel = true_coord.coord_mm_to_pixel_conversion(Y_lens)
    X_ideal_pixel = true_coord.coord_mm_to_pixel_conversion(X_ideal)
    Y_ideal_pixel = true_coord.coord_mm_to_pixel_conversion(Y_ideal)
    X_mock_pixel = true_coord.coord_mm_to_pixel_conversion(X_mock)
    Y_mock_pixel = true_coord.coord_mm_to_pixel_conversion(Y_mock)

    visualise.plot_scatter([
        [true_coord.X_true_coord_pixel, true_coord.Y_true_coord_pixel],
        [X_filtered_true_coord_pixel, Y_filtered_true_coord_pixel],
        [X_lens_pixel, Y_lens_pixel],
        [X_ideal_pixel, Y_ideal_pixel],
        [X_mock_pixel, Y_mock_pixel]
    ])

    true_coord.callibration(X_mock_pixel,Y_mock_pixel)


    print('callibration complete')