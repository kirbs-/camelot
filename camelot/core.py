# -*- coding: utf-8 -*-

import os
import sqlite3
import zipfile
import tempfile
from itertools import chain
from operator import itemgetter

import numpy as np
import pandas as pd
from pdfminer.layout import LTTextContainer


# minimum number of vertical textline intersections for a textedge
# to be considered valid
TEXTEDGE_REQUIRED_ELEMENTS = 4
# padding added to table area on the left, right and bottom
TABLE_AREA_PADDING = 10


class TextEdge(object):
    """Defines a text edge coordinates relative to a left-bottom
    origin. (PDF coordinate space)

    Parameters
    ----------
    x : float
        x-coordinate of the text edge.
    y0 : float
        y-coordinate of bottommost point.
    y1 : float
        y-coordinate of topmost point.
    align : string, optional (default: 'left')
        {'left', 'right', 'middle'}

    Attributes
    ----------
    intersections: int
        Number of intersections with horizontal text rows.
    is_valid: bool
        A text edge is valid if it intersections with at least
        TEXTEDGE_REQUIRED_ELEMENTS horizontal text rows.

    """

    def __init__(self, x, y0, y1, align="left"):
        self.x = x
        self.y0 = y0
        self.y1 = y1
        self.align = align
        self.intersections = 0
        self.is_valid = False

    def __repr__(self):
        x = round(self.x, 2)
        y0 = round(self.y0, 2)
        y1 = round(self.y1, 2)
        return f"<TextEdge x={x} y0={y0} y1={y1} align={self.align} valid={self.is_valid}>"

    def update_coords(self, x, y0, edge_tol=50):
        """Updates the text edge's x and bottom y coordinates and sets
        the is_valid attribute.
        """
        if np.isclose(self.y0, y0, atol=edge_tol):
            self.x = (self.intersections * self.x + x) / float(self.intersections + 1)
            self.y0 = y0
            self.intersections += 1
            # a textedge is valid only if it extends uninterrupted
            # over a required number of textlines
            if self.intersections > TEXTEDGE_REQUIRED_ELEMENTS:
                self.is_valid = True


class TextEdges(object):
    """Defines a dict of left, right and middle text edges found on
    the PDF page. The dict has three keys based on the alignments,
    and each key's value is a list of camelot.core.TextEdge objects.
    """

    def __init__(self, edge_tol=50):
        self.edge_tol = edge_tol
        self._textedges = {"left": [], "right": [], "middle": []}

    @staticmethod
    def get_x_coord(textline, align):
        """Returns the x coordinate of a text row based on the
        specified alignment.
        """
        x_left = textline.x0
        x_right = textline.x1
        x_middle = x_left + (x_right - x_left) / 2.0
        x_coord = {"left": x_left, "middle": x_middle, "right": x_right}
        return x_coord[align]

    def find(self, x_coord, align):
        """Returns the index of an existing text edge using
        the specified x coordinate and alignment.
        """
        for i, te in enumerate(self._textedges[align]):
            if np.isclose(te.x, x_coord, atol=0.5):
                return i
        return None

    def add(self, textline, align):
        """Adds a new text edge to the current dict.
        """
        x = self.get_x_coord(textline, align)
        y0 = textline.y0
        y1 = textline.y1
        te = TextEdge(x, y0, y1, align=align)
        self._textedges[align].append(te)

    def update(self, textline):
        """Updates an existing text edge in the current dict.
        """
        for align in ["left", "right", "middle"]:
            x_coord = self.get_x_coord(textline, align)
            idx = self.find(x_coord, align)
            if idx is None:
                self.add(textline, align)
            else:
                self._textedges[align][idx].update_coords(
                    x_coord, textline.y0, edge_tol=self.edge_tol
                )

    def generate(self, textlines):
        """Generates the text edges dict based on horizontal text
        rows.
        """
        for tl in textlines:
            if len(tl.get_text().strip()) > 1:  # TODO: hacky
                self.update(tl)

    def get_relevant(self):
        """Returns the list of relevant text edges (all share the same
        alignment) based on which list intersects horizontal text rows
        the most.
        """
        intersections_sum = {
            "left": sum(
                te.intersections for te in self._textedges["left"] if te.is_valid
            ),
            "right": sum(
                te.intersections for te in self._textedges["right"] if te.is_valid
            ),
            "middle": sum(
                te.intersections for te in self._textedges["middle"] if te.is_valid
            ),
        }

        # TODO: naive
        # get vertical textedges that intersect maximum number of
        # times with horizontal textlines
        relevant_align = max(intersections_sum.items(), key=itemgetter(1))[0]
        return self._textedges[relevant_align]

    def get_table_areas(self, textlines, relevant_textedges):
        """Returns a dict of interesting table areas on the PDF page
        calculated using relevant text edges.
        """

        def pad(area, average_row_height):
            x0 = area[0] - TABLE_AREA_PADDING
            y0 = area[1] - TABLE_AREA_PADDING
            x1 = area[2] + TABLE_AREA_PADDING
            # add a constant since table headers can be relatively up
            y1 = area[3] + average_row_height * 5
            return (x0, y0, x1, y1)

        # sort relevant textedges in reading order
        relevant_textedges.sort(key=lambda te: (-te.y0, te.x))

        table_areas = {}
        for te in relevant_textedges:
            if te.is_valid:
                if not table_areas:
                    table_areas[(te.x, te.y0, te.x, te.y1)] = None
                else:
                    found = None
                    for area in table_areas:
                        # check for overlap
                        if te.y1 >= area[1] and te.y0 <= area[3]:
                            found = area
                            break
                    if found is None:
                        table_areas[(te.x, te.y0, te.x, te.y1)] = None
                    else:
                        table_areas.pop(found)
                        updated_area = (
                            found[0],
                            min(te.y0, found[1]),
                            max(found[2], te.x),
                            max(found[3], te.y1),
                        )
                        table_areas[updated_area] = None

        # extend table areas based on textlines that overlap
        # vertically. it's possible that these textlines were
        # eliminated during textedges generation since numbers and
        # chars/words/sentences are often aligned differently.
        # drawback: table areas that have paragraphs on their sides
        # will include the paragraphs too.
        sum_textline_height = 0
        for tl in textlines:
            sum_textline_height += tl.y1 - tl.y0
            found = None
            for area in table_areas:
                # check for overlap
                if tl.y0 >= area[1] and tl.y1 <= area[3]:
                    found = area
                    break
            if found is not None:
                table_areas.pop(found)
                updated_area = (
                    min(tl.x0, found[0]),
                    min(tl.y0, found[1]),
                    max(found[2], tl.x1),
                    max(found[3], tl.y1),
                )
                table_areas[updated_area] = None
        average_textline_height = sum_textline_height / float(len(textlines))

        # add some padding to table areas
        table_areas_padded = {}
        for area in table_areas:
            table_areas_padded[pad(area, average_textline_height)] = None

        return table_areas_padded


class Cell(object):
    """Defines a cell in a table with coordinates relative to a
    left-bottom origin. (PDF coordinate space)

    Parameters
    ----------
    x1 : float
        x-coordinate of left-bottom point.
    y1 : float
        y-coordinate of left-bottom point.
    x2 : float
        x-coordinate of right-top point.
    y2 : float
        y-coordinate of right-top point.

    Attributes
    ----------
    lb : tuple
        Tuple representing left-bottom coordinates.
    lt : tuple
        Tuple representing left-top coordinates.
    rb : tuple
        Tuple representing right-bottom coordinates.
    rt : tuple
        Tuple representing right-top coordinates.
    left : bool
        Whether or not cell is bounded on the left.
    right : bool
        Whether or not cell is bounded on the right.
    top : bool
        Whether or not cell is bounded on the top.
    bottom : bool
        Whether or not cell is bounded on the bottom.
    hspan : bool
        Whether or not cell spans horizontally.
    vspan : bool
        Whether or not cell spans vertically.
    text : string
        Text assigned to cell.

    """

    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.lb = (x1, y1)
        self.lt = (x1, y2)
        self.rb = (x2, y1)
        self.rt = (x2, y2)
        self.left = False
        self.right = False
        self.top = False
        self.bottom = False
        self.hspan = False
        self.vspan = False
        self._text = ""

    def __repr__(self):
        x1 = round(self.x1, 2)
        y1 = round(self.y1, 2)
        x2 = round(self.x2, 2)
        y2 = round(self.y2, 2)
        return f"<Cell x1={x1} y1={y1} x2={x2} y2={y2}>"

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, t):
        self._text = "".join([self._text, t])

    @property
    def bound(self):
        """The number of sides on which the cell is bounded.
        """
        return self.top + self.bottom + self.left + self.right


class Table(object):
    """Defines a table with coordinates relative to a left-bottom
    origin. (PDF coordinate space)

    Parameters
    ----------
    cols : list
        List of tuples representing column x-coordinates in increasing
        order.
    rows : list
        List of tuples representing row y-coordinates in decreasing
        order.

    Attributes
    ----------
    df : :class:`pandas.DataFrame`
    shape : tuple
        Shape of the table.
    accuracy : float
        Accuracy with which text was assigned to the cell.
    whitespace : float
        Percentage of whitespace in the table.
    order : int
        Table number on PDF page.
    page : int
        PDF page number.

    """

    def __init__(self, cols, rows):
        self.cols = cols
        self.rows = rows
        self.cells = [[Cell(c[0], r[1], c[1], r[0]) for c in cols] for r in rows]
        self.df = None
        self.shape = (0, 0)
        self.accuracy = 0
        self.whitespace = 0
        self.order = None
        self.page = None

    def __repr__(self):
        return f"<{self.__class__.__name__} shape={self.shape} {','.join(self._bbox)}>"

    def __lt__(self, other):
        if self.page == other.page:
            if self.order < other.order:
                return True
        if self.page < other.page:
            return True

    @property
    def data(self):
        """Returns two-dimensional list of strings in table.
        """
        d = []
        for row in self.cells:
            d.append([cell.text.strip() for cell in row])
        return d

    @property
    def parsing_report(self):
        """Returns a parsing report with %accuracy, %whitespace,
        table number on page and page number.
        """
        # pretty?
        report = {
            "accuracy": round(self.accuracy, 2),
            "whitespace": round(self.whitespace, 2),
            "order": self.order,
            "page": self.page,
        }
        return report

    def set_all_edges(self):
        """Sets all table edges to True.
        """
        for row in self.cells:
            for cell in row:
                cell.left = cell.right = cell.top = cell.bottom = True
        return self

    def set_edges(self, vertical, horizontal, joint_tol=2):
        """Sets a cell's edges to True depending on whether the cell's
        coordinates overlap with the line's coordinates within a
        tolerance.

        Parameters
        ----------
        vertical : list
            List of detected vertical lines.
        horizontal : list
            List of detected horizontal lines.

        """
        for v in vertical:
            # find closest x coord
            # iterate over y coords and find closest start and end points
            i = [
                i
                for i, t in enumerate(self.cols)
                if np.isclose(v[0], t[0], atol=joint_tol)
            ]
            j = [
                j
                for j, t in enumerate(self.rows)
                if np.isclose(v[3], t[0], atol=joint_tol)
            ]
            k = [
                k
                for k, t in enumerate(self.rows)
                if np.isclose(v[1], t[0], atol=joint_tol)
            ]
            if not j:
                continue
            J = j[0]
            if i == [0]:  # only left edge
                L = i[0]
                if k:
                    K = k[0]
                    while J < K:
                        self.cells[J][L].left = True
                        J += 1
                else:
                    K = len(self.rows)
                    while J < K:
                        self.cells[J][L].left = True
                        J += 1
            elif i == []:  # only right edge
                L = len(self.cols) - 1
                if k:
                    K = k[0]
                    while J < K:
                        self.cells[J][L].right = True
                        J += 1
                else:
                    K = len(self.rows)
                    while J < K:
                        self.cells[J][L].right = True
                        J += 1
            else:  # both left and right edges
                L = i[0]
                if k:
                    K = k[0]
                    while J < K:
                        self.cells[J][L].left = True
                        self.cells[J][L - 1].right = True
                        J += 1
                else:
                    K = len(self.rows)
                    while J < K:
                        self.cells[J][L].left = True
                        self.cells[J][L - 1].right = True
                        J += 1

        for h in horizontal:
            # find closest y coord
            # iterate over x coords and find closest start and end points
            i = [
                i
                for i, t in enumerate(self.rows)
                if np.isclose(h[1], t[0], atol=joint_tol)
            ]
            j = [
                j
                for j, t in enumerate(self.cols)
                if np.isclose(h[0], t[0], atol=joint_tol)
            ]
            k = [
                k
                for k, t in enumerate(self.cols)
                if np.isclose(h[2], t[0], atol=joint_tol)
            ]
            if not j:
                continue
            J = j[0]
            if i == [0]:  # only top edge
                L = i[0]
                if k:
                    K = k[0]
                    while J < K:
                        self.cells[L][J].top = True
                        J += 1
                else:
                    K = len(self.cols)
                    while J < K:
                        self.cells[L][J].top = True
                        J += 1
            elif i == []:  # only bottom edge
                L = len(self.rows) - 1
                if k:
                    K = k[0]
                    while J < K:
                        self.cells[L][J].bottom = True
                        J += 1
                else:
                    K = len(self.cols)
                    while J < K:
                        self.cells[L][J].bottom = True
                        J += 1
            else:  # both top and bottom edges
                L = i[0]
                if k:
                    K = k[0]
                    while J < K:
                        self.cells[L][J].top = True
                        self.cells[L - 1][J].bottom = True
                        J += 1
                else:
                    K = len(self.cols)
                    while J < K:
                        self.cells[L][J].top = True
                        self.cells[L - 1][J].bottom = True
                        J += 1

        return self

    def set_border(self):
        """Sets table border edges to True.
        """
        for r in range(len(self.rows)):
            self.cells[r][0].left = True
            self.cells[r][len(self.cols) - 1].right = True
        for c in range(len(self.cols)):
            self.cells[0][c].top = True
            self.cells[len(self.rows) - 1][c].bottom = True
        return self

    def set_span(self):
        """Sets a cell's hspan or vspan attribute to True depending
        on whether the cell spans horizontally or vertically.
        """
        for row in self.cells:
            for cell in row:
                left = cell.left
                right = cell.right
                top = cell.top
                bottom = cell.bottom
                if cell.bound == 4:
                    continue
                elif cell.bound == 3:
                    if not left and (right and top and bottom):
                        cell.hspan = True
                    elif not right and (left and top and bottom):
                        cell.hspan = True
                    elif not top and (left and right and bottom):
                        cell.vspan = True
                    elif not bottom and (left and right and top):
                        cell.vspan = True
                elif cell.bound == 2:
                    if left and right and (not top and not bottom):
                        cell.vspan = True
                    elif top and bottom and (not left and not right):
                        cell.hspan = True
                elif cell.bound in [0, 1]:
                    cell.vspan = True
                    cell.hspan = True
        return self

    def to_csv(self, path, **kwargs):
        """Writes Table to a comma-separated values (csv) file.

        For kwargs, check :meth:`pandas.DataFrame.to_csv`.

        Parameters
        ----------
        path : str
            Output filepath.

        """
        kw = {"encoding": "utf-8", "index": False, "header": False, "quoting": 1}
        kw.update(kwargs)
        self.df.to_csv(path, **kw)

    def to_json(self, path, **kwargs):
        """Writes Table to a JSON file.

        For kwargs, check :meth:`pandas.DataFrame.to_json`.

        Parameters
        ----------
        path : str
            Output filepath.

        """
        kw = {"orient": "records"}
        kw.update(kwargs)
        json_string = self.df.to_json(**kw)
        with open(path, "w") as f:
            f.write(json_string)

    def to_excel(self, path, **kwargs):
        """Writes Table to an Excel file.

        For kwargs, check :meth:`pandas.DataFrame.to_excel`.

        Parameters
        ----------
        path : str
            Output filepath.

        """
        kw = {
            "sheet_name": f"page-{self.page}-table-{self.order}",
            "encoding": "utf-8",
        }
        kw.update(kwargs)
        writer = pd.ExcelWriter(path)
        self.df.to_excel(writer, **kw)
        writer.save()

    def to_html(self, path, **kwargs):
        """Writes Table to an HTML file.

        For kwargs, check :meth:`pandas.DataFrame.to_html`.

        Parameters
        ----------
        path : str
            Output filepath.

        """
        html_string = self.df.to_html(**kwargs)
        with open(path, "w") as f:
            f.write(html_string)

    def to_sqlite(self, path, **kwargs):
        """Writes Table to sqlite database.

        For kwargs, check :meth:`pandas.DataFrame.to_sql`.

        Parameters
        ----------
        path : str
            Output filepath.

        """
        kw = {"if_exists": "replace", "index": False}
        kw.update(kwargs)
        conn = sqlite3.connect(path)
        table_name = f"page-{self.page}-table-{self.order}"
        self.df.to_sql(table_name, conn, **kw)
        conn.commit()
        conn.close()


class TableList(object):
    """Defines a list of camelot.core.Table objects. Each table can
    be accessed using its index.

    Attributes
    ----------
    n : int
        Number of tables in the list.

    """

    def __init__(self, tables):
        self._tables = tables

    def __repr__(self):
        return f"<{self.__class__.__name__} n={self.n}>"

    def __len__(self):
        return len(self._tables)

    def __getitem__(self, idx):
        return self._tables[idx]

    @staticmethod
    def _format_func(table, f):
        return getattr(table, f"to_{f}")

    @property
    def n(self):
        return len(self)

    def _write_file(self, f=None, **kwargs):
        dirname = kwargs.get("dirname")
        root = kwargs.get("root")
        ext = kwargs.get("ext")
        for table in self._tables:
            filename = f"{root}-page-{table.page}-table-{table.order}{ext}"
            filepath = os.path.join(dirname, filename)
            to_format = self._format_func(table, f)
            to_format(filepath)

    def _compress_dir(self, **kwargs):
        path = kwargs.get("path")
        dirname = kwargs.get("dirname")
        root = kwargs.get("root")
        ext = kwargs.get("ext")
        zipname = os.path.join(os.path.dirname(path), root) + ".zip"
        with zipfile.ZipFile(zipname, "w", allowZip64=True) as z:
            for table in self._tables:
                filename = f"{root}-page-{table.page}-table-{table.order}{ext}"
                filepath = os.path.join(dirname, filename)
                z.write(filepath, os.path.basename(filepath))

    def export(self, path, f="csv", compress=False):
        """Exports the list of tables to specified file format.

        Parameters
        ----------
        path : str
            Output filepath.
        f : str
            File format. Can be csv, json, excel, html and sqlite.
        compress : bool
            Whether or not to add files to a ZIP archive.

        """
        dirname = os.path.dirname(path)
        basename = os.path.basename(path)
        root, ext = os.path.splitext(basename)
        if compress:
            dirname = tempfile.mkdtemp()

        kwargs = {"path": path, "dirname": dirname, "root": root, "ext": ext}

        if f in ["csv", "json", "html"]:
            self._write_file(f=f, **kwargs)
            if compress:
                self._compress_dir(**kwargs)
        elif f == "excel":
            filepath = os.path.join(dirname, basename)
            writer = pd.ExcelWriter(filepath)
            for table in self._tables:
                sheet_name = f"page-{table.page}-table-{table.order}"
                table.df.to_excel(writer, sheet_name=sheet_name, encoding="utf-8")
            writer.save()
            if compress:
                zipname = os.path.join(os.path.dirname(path), root) + ".zip"
                with zipfile.ZipFile(zipname, "w", allowZip64=True) as z:
                    z.write(filepath, os.path.basename(filepath))
        elif f == "sqlite":
            filepath = os.path.join(dirname, basename)
            for table in self._tables:
                table.to_sqlite(filepath)
            if compress:
                zipname = os.path.join(os.path.dirname(path), root) + ".zip"
                with zipfile.ZipFile(zipname, "w", allowZip64=True) as z:
                    z.write(filepath, os.path.basename(filepath))


class Page:
    """Defines a collection of pdfminer LTTextContainers and camelot.core.Tables.

    

    """
    def __init__(self, page_number, parser, tables):
        self.number = page_number
        self.parser = parser
        self.layout = parser.layout
        self.tables = tables
        self.text_elements = self.find_text_elements()
        
    def find_text_elements(self):
        """Identifies any text containers outside of any Camelot table bounding boxes.

        Returns:
            list: LTTextConainers outside of bounding boxes of any tables Camelot identifies.
        """
        blocks = []
        for ele in self.layout._objs:
            if isinstance(ele, LTTextContainer) and get_iou(self.tables[0]._bbox, ele.bbox) == 0.0:
                blocks.append(ele)
        return blocks
    
    def _sort(self, ele):
        """Helper to sort LTTextContainer and Tables according to y coordinate positionon a page."""
        if isinstance(ele, LTTextContainer):
            return ele.y0
        elif isinstance(ele, Table):
            return ele._bbox[1]
    
    @property
    def elements(self):
        """Combines text elements and tables on page in display order.
        
        Returns:
            list: LTTextContainer and Table elements sorted by y0 position on page.
        """
        return sorted(self.text_elements + self.tables, key=self._sort, reverse=True)
    
    @property
    def data(self):
        """Extracts text for text container and tables elements on page in display order.
        
        Returns:
            list: Extract text from elements sorted by y0 position on page.
        """
        out = []
        for ele in self.elements:
            if isinstance(ele, LTTextContainer):
                out.append(ele.get_text())
            elif isinstance(ele, Table):
                out.append(ele.data)
        return out


def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    bb1 = {
        "x1": bb1[0],
        "y1": bb1[1],
        "x2": bb1[2],
        "y2": bb1[3]
    }
    
    bb2 = {
        "x1": bb2[0],
        "y1": bb2[1],
        "x2": bb2[2],
        "y2": bb2[3]
    }
    
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou