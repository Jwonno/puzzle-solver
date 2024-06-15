import numpy as np
import os
import sys
from PIL import Image, ImageDraw
from color_labeller import label_and_color_pieces

t_FEMALE = 1
t_MALE = 2
t_LINE = 3


def computeBezierPoint(points, t):
    t_squared = t * t
    t_cubed = t_squared * t

    cx = 3.0 * (points[1][0] - points[0][0])
    bx = 3.0 * (points[2][0] - points[1][0]) - cx
    ax = points[3][0] - points[0][0] - cx - bx

    cy = 3.0 * (points[1][1] - points[0][1])
    by = 3.0 * (points[2][1] - points[1][1]) - cy
    ay = points[3][1] - points[0][1] - cy - by

    x = (ax * t_cubed) + (bx * t_squared) + (cx * t) + points[0][0]
    y = (ay * t_cubed) + (by * t_squared) + (cy * t) + points[0][1]

    return (x, y)


def computerBezier(points, num):
    dt = 1.0 / num
    curve_points = []

    for i in range(num):
        p = computeBezierPoint(points, dt * i)
        curve_points.append(p)

    return curve_points


def polygonCropImage(im, polygon, name):
    im_array = np.asarray(im)

    # create mask
    mask_im = Image.new('L', (im_array.shape[1], im_array.shape[0]), 0)
    ImageDraw.Draw(mask_im).polygon(polygon, outline=1, fill=1)
    mask = np.array(mask_im)

    # assemble new image (uint8: 0-255)
    new_im_array = np.empty(im_array.shape, dtype='uint8')

    # colors (three first columns, RGB)
    new_im_array[:, :, :3] = im_array[:, :, :3]

    # transparency (4th column)
    new_im_array[:, :, 3] = mask * 255

    # back to Image from numpy
    new_im = Image.fromarray(new_im_array, "RGBA")
    new_im.save(name)


class PieceOutLine():
    def __init__(self, width, height, ar, cr):
        self.w = width
        self.h = height
        self.arcRatio = ar
        self.connectRatio = cr
        self.pointNum = 300

    def genRightFemaleArc(self, is_top):
        half_w = self.w * 0.5
        half_h = self.h * 0.5
        arc_w = self.w * self.arcRatio
        connect_w = self.h * self.connectRatio

        top = (half_w, -half_h)
        bottom = (half_w + arc_w, -connect_w * 0.5)
        dw = bottom[0] - top[0]
        dh = bottom[1] - top[1]

        points = [top, (top[0] + dw / 3 + 8, top[1] + dh / 3), (top[0] + 2 * dw / 3 + 8, top[1] + dh * 2 / 3), bottom]

        curv_points = []
        if not is_top:
            curv_points = computerBezier(points, self.pointNum)
        else:
            left = []
            for p in reversed(points):
                left.append((p[0], -p[1]))
            curv_points = computerBezier(left, self.pointNum)

        return curv_points

    def genRightFemaleConnect(self, left):
        half_w = self.w * 0.5
        half_h = self.h * 0.5
        arc_w = self.w * self.arcRatio
        connect_w = self.h * self.connectRatio

        start_x = half_w + arc_w
        start_y = -connect_w * 0.5

        end_x = start_x - connect_w
        end_y = 0
        points = [(start_x, start_y), (end_x + (start_x - end_x) * 3 / 5, -start_y * 2 / 3), (end_x, 2 * start_y),
                  (end_x, end_y)]

        curv_points = []
        if not left:
            curv_points = computerBezier(points, self.pointNum)
        else:
            left = []
            for p in reversed(points):
                left.append((p[0], -p[1]))
            curv_points = computerBezier(left, self.pointNum)

        return curv_points

    def genBottomFemaleArc(self, is_left):
        half_w = self.w * 0.5
        half_h = self.h * 0.5
        arc_h = self.h * self.arcRatio
        connect_w = self.w * self.connectRatio

        right = (half_w, half_h)
        left = (connect_w * 0.5, half_h + arc_h)
        dw = right[0] - left[0]
        dh = left[1] - right[1]

        points = [right, (right[0] - dw / 3, right[1] + dh / 3 + 8), (right[0] - 2 * dw / 3, right[1] + dh * 2 / 3 + 8),
                  left]

        curv_points = []
        if not is_left:
            curv_points = computerBezier(points, self.pointNum)
        else:
            left = []
            for p in reversed(points):
                left.append((-p[0], p[1]))
            curv_points = computerBezier(left, self.pointNum)

        return curv_points

    def genBottomFemaleConnect(self, left):
        half_w = self.w * 0.5
        half_h = self.h * 0.5
        arc_h = self.h * self.arcRatio
        connect_w = self.w * self.connectRatio

        start_x = connect_w * 0.5
        start_y = half_h + arc_h

        end_x = 0
        end_y = start_y - connect_w
        points = [(start_x, start_y), (-start_x * 2 / 3, end_y + (start_y - end_y) * 3 / 5), (2 * start_x, end_y),
                  (end_x, end_y)]

        curv_points = []
        if not left:
            curv_points = computerBezier(points, self.pointNum)
        else:
            left = []
            for p in reversed(points):
                left.append((-p[0], p[1]))
            curv_points = computerBezier(left, self.pointNum)

        return curv_points

    def genRightFemale(self):
        right_arc = self.genRightFemaleArc(False)
        right = self.genRightFemaleConnect(False)
        left = self.genRightFemaleConnect(True)
        left_arc = self.genRightFemaleArc(True)

        curv_points = right_arc + right + left + left_arc
        return curv_points

    def genRightMale(self):
        half_w = self.w * 0.5
        points = self.genRightFemale()
        curv_points = []
        for p in points:
            curv_points.append((p[0] + (half_w - p[0]) * 2, p[1]))
        return curv_points

    def genRightLine(self):
        return [(self.w * 0.5, self.h * 0.5)]

    def genLeftMale(self):
        half_w = self.w * 0.5
        points = self.genRightFemale()
        curv_points = []
        for p in points:
            curv_points.append((p[0] - half_w * 2, p[1]))
        return reversed(curv_points)

    def genLeftFemale(self):
        half_w = self.w * 0.5
        points = self.genLeftMale()
        curv_points = []
        for p in points:
            curv_points.append(((-p[0] - half_w) * 2 + p[0], p[1]))
        return curv_points

    def genLeftLine(self):
        return [(-self.w * 0.5, -self.h * 0.5)]

    def genBottomFemale(self):
        right_arc = self.genBottomFemaleArc(False)
        right = self.genBottomFemaleConnect(False)
        left = self.genBottomFemaleConnect(True)
        left_arc = self.genBottomFemaleArc(True)

        curv_points = right_arc + right + left + left_arc
        return curv_points

    def genBottomMale(self):
        half_h = self.h * 0.5
        points = self.genBottomFemale()
        curv_points = []
        for p in points:
            curv_points.append((p[0], (half_h - p[1]) * 2 + p[1]))
        return curv_points

    def genBottomLine(self):
        return [(-self.w * 0.5, self.h * 0.5)]

    def genTopMale(self):
        points = self.genBottomFemale()
        curv_points = []
        for p in points:
            curv_points.append((p[0], p[1] - self.h))

        return reversed(curv_points)

    def genTopFemale(self):
        half_h = self.h * 0.5
        points = self.genTopMale()
        curv_points = []
        for p in points:
            curv_points.append((p[0], (-p[1] - half_h) * 2 + p[1]))
        return curv_points

    def genTopLine(self):
        return [(self.w * 0.5, -self.h * 0.5)]

    def genOutLine(self, piece_borders):
        curv_points = []
        curv_points.append((self.w * 0.5, self.h * 0.5))
        func = [
            {
                t_FEMALE: self.genBottomFemale,
                t_MALE: self.genBottomMale,
                t_LINE: self.genBottomLine,
            },
            {
                t_FEMALE: self.genLeftFemale,
                t_MALE: self.genLeftMale,
                t_LINE: self.genLeftLine,
            },
            {
                t_FEMALE: self.genTopFemale,
                t_MALE: self.genTopMale,
                t_LINE: self.genTopLine,
            },
            {
                t_FEMALE: self.genRightFemale,
                t_MALE: self.genRightMale,
                t_LINE: self.genRightLine,
            },
        ]

        for i, f in enumerate(func):
            curv_points += f[piece_borders[i]]()

        return curv_points


class PieceInfo():
    def __init__(self, size, row_num, col_num, ar, cr):
        self.w = size[0] / col_num
        self.h = size[1] / row_num
        self.row_num = row_num
        self.col_num = col_num
        self.arc_ratio = ar
        self.connect_ratio = cr

    def getPieceInfo(self, row, col):
        arc_w = self.w * self.arc_ratio
        arc_h = self.h * self.arc_ratio
        connect_w = self.w * self.connect_ratio
        connect_h = self.h * self.connect_ratio
        borders = []

        t = t_MALE

        if (row + col) % 2 == 0:
            t = t_FEMALE

        if t == t_FEMALE:
            borders = [t_MALE, t_FEMALE, t_MALE, t_FEMALE]
        else:
            borders = [t_FEMALE, t_MALE, t_FEMALE, t_MALE]

        if col == 0:
            borders[1] = t_LINE

        if row == 0:
            borders[2] = t_LINE

        if (row + 1) == self.row_num:
            borders[0] = t_LINE

        if (col + 1) == self.col_num:
            borders[3] = t_LINE

        top_x = self.w * col
        top_y = self.h * row
        bottom_x = self.w * (col + 1)
        bottom_y = self.h * (row + 1)
        center_x = self.w * 0.5
        center_y = self.h * 0.5

        # Bottom
        if borders[0] == t_MALE:
            bottom_y += (connect_w - arc_h)
        elif borders[0] == t_FEMALE:
            bottom_y += arc_h

        # Left
        if borders[1] == t_MALE:
            top_x -= (connect_h - arc_w)
            center_x += (connect_h - arc_w)
        elif borders[1] == t_FEMALE:
            top_x -= arc_w
            center_x += arc_w
        # Top
        if borders[2] == t_MALE:
            top_y -= (connect_w - arc_h)
            center_y += (connect_w - arc_h)
        elif borders[2] == t_FEMALE:
            top_y -= arc_h
            center_y += arc_h

        # Right
        if borders[3] == t_MALE:
            bottom_x += (connect_h - arc_w)
        elif borders[3] == t_FEMALE:
            bottom_x += arc_w

        return (
            int(round(top_x)),
            int(round(top_y)),
            int(round(bottom_x)),
            int(round(bottom_y))
        ), (center_x, center_y), borders


def createPuzzlePieces(name, row, col, out_prefix):
    im = Image.open(name).convert("RGBA")
    arc_ratio = 0.07
    connect_ratio = 0.3
    r = 1  # Line thickness control

    info = PieceInfo(im.size, row, col, arc_ratio, connect_ratio)

    outLine = PieceOutLine(im.size[0] / col, im.size[1] / row, arc_ratio, connect_ratio)

    w = im.size[0] / col
    h = im.size[1] / row

    draw = ImageDraw.Draw(im)
    outLinePoints = []
    json = "{"
    first = True
    for i in range(row):
        for j in range(col):
            rect, center, borders = info.getPieceInfo(i, j)
            piece_name = out_prefix + str(i) + "_" + str(j)
            if not first:
                json += ","
            first = False

            json += f'\n    "{os.path.basename(piece_name)}":[{rect[0]},{rect[1]}]'
            region = im.crop(rect)
            curv_points = outLine.genOutLine(borders)

            crop_points = []
            for p in curv_points:
                crop_points.append((p[0] + center[0], p[1] + center[1]))
            polygonCropImage(region, crop_points, piece_name + ".png")

            for p in curv_points:
                outLinePoints.append((p[0] + j * w + 0.5 * w, p[1] + i * h + 0.5 * h))

    json += "\n}\n"
    with open(out_prefix + "data.json", "w") as dataFile:
        dataFile.write(json)

    bg_color = (255, 255, 255, 0)
    line_color = (0, 0, 0, 255)

    outLinedraw = ImageDraw.Draw(im)

    outLinedraw.rectangle([(0, 0), (im.size[0], im.size[1])], fill=bg_color, outline=bg_color)

    for p in outLinePoints:
        px = p[0]
        py = p[1]
        outLinedraw.ellipse((px - r, py - r, px + r, py + r), fill=line_color, outline=line_color)

    for x in range(im.size[0]):
        px = x
        py = 0.5 * r
        outLinedraw.ellipse((px - r, py - r, px + r, py + r), fill=line_color, outline=line_color)
        py = im.size[1] - 0.5 * r
        outLinedraw.ellipse((px - r, py - r, px + r, py + r), fill=line_color, outline=line_color)

    for y in range(im.size[1]):
        px = 0.5 * r
        py = y
        outLinedraw.ellipse((px - r, py - r, px + r, py + r), fill=line_color, outline=line_color)
        px = im.size[0] - 0.5 * r
        outLinedraw.ellipse((px - r, py - r, px + r, py + r), fill=line_color, outline=line_color)

    im.save(out_prefix + "outline.png")


def main(image_path, rows, columns):
    out_dir = "test_pieces/" + os.path.splitext(os.path.basename(image_path))[0] + "_{}".format(rows)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    out_prefix = out_dir + "/piece_"
    createPuzzlePieces(image_path, rows, columns, out_prefix)

    # Label and color puzzle pieces
    outline_path = out_prefix + "outline.png"
    color_output_path = out_prefix + "colored_regions.png"
    label_output_path = out_prefix + "labeled_regions.png"
    csv_output_path = out_prefix + "puzzle_colors.csv"
    label_and_color_pieces(outline_path, color_output_path, label_output_path, csv_output_path)

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(f"Usage: {sys.argv[0]} image_path rows columns")
    else:
        main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
