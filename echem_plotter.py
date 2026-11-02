"""电化学论文曲线图生成器 / 電気化学 論文用グラフ作成ツール

运行 / 実行:
    pip install streamlit numpy pandas scipy matplotlib openpyxl google-genai plotly
    streamlit run echem_plotter.py
"""
import hashlib
import io
import json
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import font_manager
from matplotlib.ticker import MultipleLocator
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, savgol_filter

try:
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

MM = 1 / 25.4
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]
CUR = {"A": 1.0, "mA": 1e-3, "µA": 1e-6, "nA": 1e-9}
REF = {"Ag/AgCl (sat. KCl)": 0.197, "SCE": 0.241, "Hg/HgO (1 M NaOH)": 0.098, "Hg/Hg₂SO₄": 0.615}
FITS = {
    "CV": ["none", "linear", "poly"],
    "CA": ["none", "cottrell", "exp", "linear", "poly"],
    "CP": ["none", "linear", "poly", "exp"],
}
MIME = {"png": "image/png", "pdf": "application/pdf", "svg": "image/svg+xml", "tiff": "image/tiff"}
SIDES = ["bottom", "left", "top", "right"]

# 字体设置
_LATIN = ["Times New Roman", "Liberation Serif", "Nimbus Roman", "TeX Gyre Termes", "STIXGeneral"]
_CJK = [
    "Yu Mincho", "MS Mincho", "MS PMincho", "Hiragino Mincho ProN", "IPAexMincho", "IPAMincho",
    "Noto Serif CJK JP", "SimSun", "NSimSun", "Songti SC", "Noto Serif CJK SC",
]
_AVAIL = {f.name for f in font_manager.fontManager.ttflist}
FONTS = [f for f in _LATIN + _CJK if f in _AVAIL] + ["DejaVu Serif"]

# ---------------------------------------------------------------- 多语言 / 多言語
TR = {
    "zh": {
        "title": "电化学论文曲线图",
        "mode_label": "曲线类型",
        "mode_CV": "CV 循环伏安",
        "mode_CA": "CA 计时电流（恒电位）",
        "mode_CP": "CP 计时电位（恒电流）",
        "upload": "上传数据（可多选，用于叠加）",
        "use_demo": "未上传文件时使用示例数据",
        "enc": "文件编码（表头出现乱码时手动选）",
        "enc_auto": "自动识别",
        "enc_utf8": "UTF-8",
        "enc_gbk": "GBK（中文 Windows）",
        "enc_sjis": "Shift-JIS（日文 Windows）",
        "info_upload": "上传仪器导出的 txt / CSV / xlsx 开始，或勾选左侧的示例数据先试试。",
        "demo_note": "当前显示的是示例数据，不是你的数据。",
        "parse_fail": "{name} 解析失败：{err}",
        "err_two_cols": "数据至少需要两列。",
        "sec_data": "数据列",
        "grp_data": "数据",
        "grp_proc": "数据处理",
        "grp_axes": "坐标轴",
        "grp_curves": "曲线与图例",
        "grp_fig": "图形外观",
        "sub_cols": "数据列",
        "sub_rows": "使用的行范围",
        "sub_curves": "曲线颜色",
        "sub_legend": "图例",
        "ax_x": "下方 x 轴",
        "ax_x2": "上方 x 轴 (x2)",
        "ax_y": "左 y 轴",
        "ax_y2": "右 y 轴 (y2)",
        "ax_color": "轴的颜色（标签、刻度、轴线）",
        "color_xb": "下方 x 轴颜色",
        "sub_ticks_common": "刻度设置（四个轴共用）",
        "color_x2curve": "{name} 上方 x 轴曲线颜色",
        "leg_text_x2": "{name} (上方 x 轴) 图例名称",
        "time_unit": "时间单位（原始数据按秒）",
        "ref_custom": "自定义…",
        "ref_custom_name": "参比电极名称（显示在轴标签里）",
        "ref_custom_E": "该电极相对 SHE 的电位 / V（换算 RHE 用）",
        "reset_mode": "剔除点后坐标递补",
        "reset_all": "全部收拢（所有被剔除的空档都补上）",
        "reset_head": "只收拢开头（第一个保留点归零，中间空档保留）",
        "reset_off": "不递补",
        "reset_note": "递补只作用于时间轴，CV 的电位不会被改动。",
        "leg_manual": "手动指定位置",
        "leg_x": "图例 x（0 左 → 1 右）",
        "leg_y": "图例 y（0 下 → 1 上）",
        "leg_manual_note": "以图例左上角为基准点，坐标是相对绘图区的比例，可以超出 0–1 把图例放到图外。",
        "leg_ncol": "图例列数",
        "leg_frame": "显示图例边框",
        "x2_note": "上方 x 轴：以所选列为横坐标，把左 y 轴的数据再画一条曲线（颜色、图例名称可单独设置）。类型选“电位”会做参比换算，选“时间”会跟随“时间单位”。",
        "xcol": "x 列",
        "ycol": "y 列（左轴）",
        "start": "起始行",
        "end": "结束行（0=到末尾）",
        "x_to_min": "时间轴换算为分钟 (÷60)",
        "sec_dual": "双轴设置 (x2 / y2)",
        "dual": "启用右侧第二个 y 轴 (y2)",
        "dual_x": "启用上方第二个 x 轴 (x2)",
        "y2col": "右轴数据列",
        "y2kind": "右轴的物理量类型",
        "x2col": "上方 x 轴数据列",
        "x2kind": "上方 x 轴物理量类型",
        "color_xt": "上方 x 轴颜色",
        "kind_cur": "电流（做单位换算）",
        "kind_pot": "电位（做参比换算）",
        "kind_time": "时间（不换算）",
        "kind_other": "其他（不换算）",
        "sec_units": "单位与参比",
        "cur_raw": "原始电流单位",
        "cur_out": "输出电流单位",
        "norm": "按电极面积归一化为电流密度",
        "area": "电极面积 / cm²",
        "ref": "参比电极",
        "ref_none": "无（不标注参比电极）",
        "to_rhe": "换算为 RHE",
        "ph": "溶液 pH",
        "sec_clean": "坏点剔除",
        "hampel": "自动检测离群点（Hampel）",
        "win": "窗口（点）",
        "hampel_note": "窗口太小、阈值太低，会把正常的数据点也当成坏点。先用默认值，再看预览里的红色 × 是否只落在真正的尖峰上。",
        "nsig": "阈值（σ）",
        "ex_range": "按 x 范围剔除（图上坐标，如 0.1:0.2; 5:6）",
        "ex_idx": "按行号剔除（如 5, 10-12）",
        "reset_idx": "剔除点后坐标/序号自动递补归零",
        "sec_smooth": "平滑",
        "sm_method": "平滑方法",
        "sm_none": "不平滑",
        "sm_savgol": "Savitzky–Golay",
        "sm_ma": "移动平均",
        "sm_gauss": "高斯滤波",
        "sm_win": "窗口（点，奇数）",
        "sm_poly": "多项式阶数",
        "sm_sigma": "σ（点）",
        "sm_right": "同时平滑右轴曲线",
        "sm_raw": "同时显示原始数据（浅色）",
        "sm_note": "拟合使用的是剔除坏点后、未平滑的原始数据。",
        "sec_fit": "拟合（针对左轴曲线）",
        "fit_model": "模型",
        "deg": "多项式阶数",
        "fit_range": "拟合范围（图上 x 坐标，如 5:60；留空为全部）",
        "fit_none": "不拟合",
        "fit_linear": "线性",
        "fit_poly": "多项式",
        "fit_exp": "指数衰减",
        "fit_cottrell": "Cottrell",
        "sec_fig": "图形",
        "width": "宽度",
        "w85": "单栏 85 mm",
        "w120": "1.5 栏 120 mm",
        "w175": "双栏 175 mm",
        "height": "高度 / mm",
        "fontsize": "字号 / pt",
        "scatter": "以散点显示数据",
        "legend": "显示图例",
        "legend_loc": "图例位置",
        "sec_range": "坐标轴范围（留空为自动）",
        "xmin": "x 最小",
        "xmax": "x 最大",
        "ymin": "y 最小",
        "ymax": "y 最大",
        "y2min": "右 y 最小",
        "y2max": "右 y 最大",
        "x2min": "上方 x2 最小",
        "x2max": "上方 x2 最大",
        "warn_num": "坐标轴范围和刻度间隔需要填写数字，间隔要大于 0。",
        "step_title": "主刻度间隔（留空为自动）",
        "stepx": "x 间隔",
        "stepy": "y 间隔",
        "stepy2": "右 y 间隔",
        "stepx2": "上方 x 间隔",
        "warn_step": "主刻度间隔太小，会产生过多刻度，这一项已被忽略。",
        "lab_note": "**上下角标的写法**：用 `$` 括住要变角标的部分，`^` 表示上标，`_` 表示下标，多个字符用 `{}` 括起来。  \n例：`cm$^{-2}$` → cm⁻²；`H$_2$O$_2$` → H₂O₂；`$E_{pa}$` → E 加下标 pa；`$\\mu$A` → µA。  \n注意：`$` 要成对，符号必须是英文半角。含汉字的标签里，角标会自动换成 Unicode 字符，大写字母的下标不支持。",
        "sec_ticks": "刻度显示",
        "tick_marks": "刻度线（显示哪几边）",
        "tick_nums": "刻度数字（显示哪几边）",
        "side_bottom": "下",
        "side_left": "左",
        "side_top": "上",
        "side_right": "右",
        "minor": "显示次刻度",
        "tdir": "刻度方向",
        "dir_in": "朝内",
        "dir_out": "朝外",
        "sec_color": "颜色",
        "color_curve": "{name} 曲线颜色",
        "color_left": "{name} 左轴曲线颜色",
        "color_right": "{name} 右轴曲线颜色",
        "color_fit": "拟合线颜色",
        "yaxis_color": "y 轴颜色（标签、刻度、轴线）",
        "yc_black": "黑色",
        "yc_follow": "跟随曲线颜色",
        "yc_custom": "自定义",
        "color_yl": "左 y 轴颜色",
        "color_yr": "右 y 轴颜色",
        "sec_leg_text": "自定义图例名称（留空为默认）",
        "leg_text_single": "{name} 图例名称",
        "leg_text_left": "{name} (左轴) 图例名称",
        "leg_text_right": "{name} (右轴) 图例名称",
        "sec_labels": "坐标轴标签（留空为自动）",
        "lab_x": "x 轴标签",
        "lab_x2": "上方 x 轴标签 (x2)",
        "lab_y": "左 y 轴标签",
        "lab_y2": "右 y 轴标签",
        "lab_help": "上标：`cm$^{-2}$`；下标：`H$_2$O$_2$`；斜体变量加下标：`$E_{pa}$`；希腊字母：`$\\mu$A`。$ 必须成对，上下标有多个字符时用 {} 括起来。含汉字时，上下标会自动换成 Unicode 字符，此时大写字母的下标不支持。",
        "sec_ai": "AI 分析",
        "ai_intro": "把处理后的数据摘要（峰位、拟合参数、统计量等）发给 Gemini，让它帮你解读。发送的是摘要，不是原始数据文件；发送前可以在下面展开查看具体内容。",
        "ai_warn": "AI 的解读可能有误，仅供参考，请对照原始数据和文献自行核对。发送的内容会经过 Google 的服务器，未发表的数据请谨慎处理。",
        "ai_key": "API 密钥",
        "ai_key_help": "只在本次使用中保存在内存里，不会写入文件。不要把密钥写进代码，也不要上传到 GitHub。",
        "ai_model": "模型",
        "ai_model_help": "填写 Google Gemini API 的模型名称。当前默认模型为 gemini-3.8-flash。",
        "ai_ctx": "实验背景（可选）",
        "ai_ctx_ph": "例如：多孔金电极，0.1 M PBS，检测葡萄糖",
        "ai_scan": "扫速 / mV s⁻¹（CV，可选）",
        "ai_q": "想问 AI 的问题（可选）",
        "ai_sendcurve": "同时发送降采样的曲线数据（每条最多 60 点）",
        "ai_preview": "查看将发送给 AI 的内容",
        "ai_btn": "生成 AI 分析",
        "ai_spin": "AI 分析中……",
        "ai_nokey": "请先在侧边栏的“AI 分析”里填写 API 密钥。",
        "ai_nopkg": "没有安装 google-genai 包，请运行：pip install google-genai",
        "ai_fail": "AI 调用失败：{err}",
        "ai_stale": "（这条分析是按之前的数据和设置生成的，改过参数后请重新生成。）",
        "sec_peaks": "CV 峰分析",
        "pk_note": "峰电流未做基线校正，单位同 y 轴；扫描方向由电位变化推断，默认氧化电流为正。使用了平滑时，峰是在平滑后的曲线上找的。",
        "pk_none": "没有检测到明显的氧化或还原峰。",
        "tab_static": "静态预览（出版样式）",
        "tab_inter": "交互查看（悬停读数）",
        "tab_table": "处理后的数据",
        "hover_mode": "悬停读数方式",
        "hv_closest": "最近的数据点",
        "hv_x": "同一 x 处的所有曲线",
        "crosshair": "显示十字准线",
        "show_removed": "显示被剔除的点（红色 ×）",
        "inter_hint": "鼠标放在曲线上会显示坐标和行号，行号可以直接填进“按行号剔除”。滚轮缩放，拖动平移，双击还原。这里只用于查看，导出的图以“静态预览”为准。",
        "no_plotly": "没有安装 plotly，请运行：pip install plotly",
        "show_peaks": "在图上标注峰位（CV）",
        "tbl_only_kept": "只保留未被剔除的点",
        "tbl_file_sel": "选择文件",
        "tbl_dl": "下载处理后的数据（CSV）",
        "tbl_hint": "这里是剔除坏点、平滑和单位换算之后的数据，可以直接导入 Origin 或 Excel。",
        "warn_range": "剔除或拟合范围的格式有误，请检查。",
        "preview_caption": "预览（红色 × 为已剔除的点，导出时不含）",
        "results": "结果",
        "tbl_file": "文件",
        "tbl_removed": "剔除点数",
        "tbl_params": "参数",
        "tbl_r2": "R²",
        "fit_fail": "拟合失败：{err}",
        "sec_export": "输出图片",
        "fmt": "格式",
        "dpi": "分辨率 (dpi)",
        "fname": "文件名",
        "dl": "下载图片",
        "dl_hint": "PDF、SVG 是矢量图，放大不会模糊；PNG、TIFF 是位图。图上英文使用 Times New Roman，日文汉字使用明朝体，需要电脑里装有对应字体。",
    },
    "ja": {
        "title": "電気化学 論文用グラフ作成ツール",
        "mode_label": "測定の種類",
        "mode_CV": "CV サイクリックボルタンメトリー",
        "mode_CA": "CA クロノアンペロメトリー（定電位）",
        "mode_CP": "CP クロノポテンショメトリー（定電流）",
        "upload": "データをアップロード（複数可・重ね描き用）",
        "use_demo": "ファイル未選択のときはサンプルデータを使う",
        "enc": "ファイルの文字コード（文字化けしたときに手動で選択）",
        "enc_auto": "自動判定",
        "enc_utf8": "UTF-8",
        "enc_gbk": "GBK（中国語 Windows）",
        "enc_sjis": "Shift-JIS（日本語 Windows）",
        "info_upload": "装置から出力した txt / CSV / xlsx をアップロードするか、左のサンプルデータを選んで試してください。",
        "demo_note": "いま表示しているのはサンプルデータで、あなたのデータではありません。",
        "parse_fail": "{name} の読み込みに失敗しました：{err}",
        "err_two_cols": "データには少なくとも2列必要です。",
        "sec_data": "データ列",
        "grp_data": "データ",
        "grp_proc": "データ処理",
        "grp_axes": "軸",
        "grp_curves": "曲線と凡例",
        "grp_fig": "図の外観",
        "sub_cols": "データ列",
        "sub_rows": "使用する行の範囲",
        "sub_curves": "曲線の色",
        "sub_legend": "凡例",
        "ax_x": "下部 x 軸",
        "ax_x2": "上部 x 軸 (x2)",
        "ax_y": "左 y 軸",
        "ax_y2": "右 y 軸 (y2)",
        "ax_color": "軸の色（ラベル・目盛り・軸線）",
        "color_xb": "下部 x 軸の色",
        "sub_ticks_common": "目盛りの設定（全軸共通）",
        "color_x2curve": "{name} 上部 x 軸の曲線の色",
        "leg_text_x2": "{name} (上部 x 軸) 凡例",
        "time_unit": "時間の単位（元データは秒）",
        "ref_custom": "カスタム…",
        "ref_custom_name": "参照電極の名称（軸ラベルに表示）",
        "ref_custom_E": "この電極の SHE 基準の電位 / V（RHE 換算用）",
        "reset_mode": "除去後の座標の詰め方",
        "reset_all": "すべて詰める（除去した隙間をすべて埋める）",
        "reset_head": "先頭だけ詰める（最初の残った点を 0 に、途中の隙間は残す）",
        "reset_off": "詰めない",
        "reset_note": "詰める処理は時間軸にだけ働きます。CV の電位は変更されません。",
        "leg_manual": "位置を手動指定",
        "leg_x": "凡例 x（0 左 → 1 右）",
        "leg_y": "凡例 y（0 下 → 1 上）",
        "leg_manual_note": "凡例の左上を基準点とし、座標は描画領域に対する比率です。0–1 の範囲外にして図の外に置くこともできます。",
        "leg_ncol": "凡例の列数",
        "leg_frame": "凡例の枠を表示",
        "x2_note": "上部 x 軸：選んだ列を横軸にして、左 y 軸のデータをもう一本描きます（色・凡例名は個別に設定できます）。種類で「電位」を選ぶと参照電極換算、「時間」を選ぶと「時間の単位」が適用されます。",
        "xcol": "x 列",
        "ycol": "y 列（左軸）",
        "start": "開始行",
        "end": "終了行（0=最後まで）",
        "x_to_min": "時間軸を分に変換 (÷60)",
        "sec_dual": "二軸の設定 (x2 / y2)",
        "dual": "右側に第2の y 軸を使う (y2)",
        "dual_x": "上部に第2の x 軸を使う (x2)",
        "y2col": "右軸のデータ列",
        "y2kind": "右軸の物理量の種類",
        "x2col": "上部 x 軸データ列",
        "x2kind": "上部 x 軸の物理量の種類",
        "color_xt": "上部 x 軸の色",
        "kind_cur": "電流（単位換算あり）",
        "kind_pot": "電位（参照電極換算あり）",
        "kind_time": "時間（換算なし）",
        "kind_other": "その他（換算しない）",
        "sec_units": "単位と参照電極",
        "cur_raw": "元の電流単位",
        "cur_out": "出力する電流単位",
        "norm": "電極面積で規格化して電流密度にする",
        "area": "電極面積 / cm²",
        "ref": "参照電極",
        "ref_none": "なし（参照電極を記載しない）",
        "to_rhe": "RHE に換算",
        "ph": "溶液の pH",
        "sec_clean": "外れ値の除去",
        "hampel": "外れ値を自動検出（Hampel）",
        "win": "ウィンドウ（点）",
        "hampel_note": "ウィンドウが小さすぎたりしきい値が低すぎたりすると、正常なデータ点まで外れ値と判定されます。まず初期値で試し、プレビューの赤い × が本当のスパイクにだけ付いているか確認してください。",
        "nsig": "しきい値（σ）",
        "ex_range": "x 範囲で除外（グラフ上の座標。例 0.1:0.2; 5:6）",
        "ex_idx": "行番号で除外（例 5, 10-12）",
        "reset_idx": "除外後、座標と行番号をゼロから再配置",
        "sec_smooth": "平滑化",
        "sm_method": "平滑化の方法",
        "sm_none": "平滑化しない",
        "sm_savgol": "Savitzky–Golay",
        "sm_ma": "移動平均",
        "sm_gauss": "ガウシアンフィルタ",
        "sm_win": "ウィンドウ幅（点・奇数）",
        "sm_poly": "多項式の次数",
        "sm_sigma": "σ（点）",
        "sm_right": "右軸の曲線も平滑化する",
        "sm_raw": "元データも薄い色で重ねて表示",
        "sm_note": "フィッティングには、外れ値を除いた平滑化前のデータを使います。",
        "sec_fit": "フィッティング（左軸の曲線が対象）",
        "fit_model": "モデル",
        "deg": "多項式の次数",
        "fit_range": "フィット範囲（グラフ上の x 座標。例 5:60、空欄で全体）",
        "fit_none": "フィットしない",
        "fit_linear": "直線",
        "fit_poly": "多項式",
        "fit_exp": "指数減衰",
        "fit_cottrell": "Cottrell",
        "sec_fig": "図の設定",
        "width": "幅",
        "w85": "1 段 85 mm",
        "w120": "1.5 段 120 mm",
        "w175": "2 段 175 mm",
        "height": "高さ / mm",
        "fontsize": "文字サイズ / pt",
        "scatter": "データを点で表示",
        "legend": "凡例を表示",
        "legend_loc": "凡例の位置",
        "sec_range": "軸の範囲（空欄で自動）",
        "xmin": "x 最小",
        "xmax": "x 最大",
        "ymin": "y 最小",
        "ymax": "y 最大",
        "y2min": "右 y 最小",
        "y2max": "右 y 最大",
        "x2min": "上部 x2 最小",
        "x2max": "上部 x2 最大",
        "warn_num": "軸の範囲と目盛りの間隔には数値を入力してください（間隔は 0 より大きい値）。",
        "step_title": "主目盛りの間隔（空欄で自動）",
        "stepx": "x 間隔",
        "stepy": "y 間隔",
        "stepy2": "右 y 間隔",
        "stepx2": "上部 x 間隔",
        "warn_step": "主目盛りの間隔が小さすぎて目盛りが多くなりすぎるため、この設定は無視しました。",
        "lab_note": "**上付き・下付きの書き方**：添字にしたい部分を `$` で囲み、`^` が上付き、`_` が下付きです。複数文字のときは `{}` で囲みます。  \n例：`cm$^{-2}$` → cm⁻²、`H$_2$O$_2$` → H₂O₂、`$E_{pa}$` → E に下付き pa、`$\\mu$A` → µA。  \n注意：`$` は必ず対にし、記号は半角の英数字を使います。漢字を含むラベルでは添字が Unicode 文字に自動変換されるため、大文字の下付きは使えません。",
        "sec_ticks": "目盛りの表示",
        "tick_marks": "目盛り線（表示する辺）",
        "tick_nums": "目盛りの数値（表示する辺）",
        "side_bottom": "下",
        "side_left": "左",
        "side_top": "上",
        "side_right": "右",
        "minor": "補助目盛りを表示",
        "tdir": "目盛りの向き",
        "dir_in": "内向き",
        "dir_out": "外向き",
        "sec_color": "色",
        "color_curve": "{name} の曲線の色",
        "color_left": "{name} 左軸の曲線の色",
        "color_right": "{name} 右軸の曲線の色",
        "color_fit": "フィット線の色",
        "yaxis_color": "y 轴の色（ラベル・目盛り・軸線）",
        "yc_black": "黒",
        "yc_follow": "曲線の色に合わせる",
        "yc_custom": "カスタム",
        "color_yl": "左 y 軸の色",
        "color_yr": "右 y 軸の色",
        "sec_leg_text": "凡例テキストの変更（空欄でデフォルト）",
        "leg_text_single": "{name} の凡例",
        "leg_text_left": "{name} (左軸) 凡例",
        "leg_text_right": "{name} (右軸) 凡例",
        "sec_labels": "軸ラベル（空欄で自動）",
        "lab_x": "x 軸ラベル",
        "lab_x2": "上部 x 軸ラベル (x2)",
        "lab_y": "左 y 軸ラベル",
        "lab_y2": "右 y 軸ラベル",
        "lab_help": "上付き：`cm$^{-2}$`、下付き：`H$_2$O$_2$`、斜体の変数と添字：`$E_{pa}$`、ギリシャ文字：`$\\mu$A`。$ は必ず対にし、添字が複数文字のときは {} で囲みます。漢字を含むラベルでは添字が Unicode 文字に自動変換されるため、大文字の下付きは使えません。",
        "sec_ai": "AI 分析",
        "ai_intro": "処理後のデータの要約（ピーク位置、フィットのパラメータ、統計量など）を Gemini に送り、解釈してもらいます。送るのは要約で、元のデータファイルではありません。送信前に下で内容を確認できます。",
        "ai_warn": "AI の解釈は誤っていることがあります。参考程度にとどめ、元のデータや文献と照らして確認してください。送信内容は Google のサーバーを経由します。未発表のデータの扱いにはご注意ください。",
        "ai_key": "API キー",
        "ai_key_help": "今回の利用中だけメモリ上に保持し、ファイルには書き込みません。キーをコードに書いたり、GitHub にアップロードしたりしないでください。",
        "ai_model": "モデル",
        "ai_model_help": "Google Gemini API のモデル名を入力します。デフォルトは gemini-3.8-flash です。",
        "ai_ctx": "実験の背景（任意）",
        "ai_ctx_ph": "例：多孔質金電極、0.1 M PBS、グルコース検出",
        "ai_scan": "掃引速度 / mV s⁻¹（CV・任意）",
        "ai_q": "AI への質問（任意）",
        "ai_sendcurve": "間引いた曲線データも送る（1 本あたり最大 60 点）",
        "ai_preview": "AI に送る内容を確認する",
        "ai_btn": "AI 分析を実行",
        "ai_spin": "AI が分析中……",
        "ai_nokey": "先にサイドバーの「AI 分析」で API キーを入力してください。",
        "ai_nopkg": "google-genai パッケージがありません。pip install google-genai を実行してください。",
        "ai_fail": "AI の呼び出しに失敗しました：{err}",
        "ai_stale": "（この分析は以前のデータと設定で作られたものです。設定を変えたら再実行してください。）",
        "sec_peaks": "CV ピーク解析",
        "pk_note": "ピーク電流はベースライン補正なしで、単位は y 軸と同じです。掃引方向は電位の変化から推定し、酸化電流を正としています。平滑化した場合は平滑化後の曲線でピークを探します。",
        "pk_none": "明確な酸化・還元ピークは検出されませんでした。",
        "tab_static": "静的プレビュー（論文用）",
        "tab_inter": "インタラクティブ表示（ホバーで読み取り）",
        "tab_table": "処理後のデータ",
        "hover_mode": "ホバーの方式",
        "hv_closest": "最も近いデータ点",
        "hv_x": "同じ x のすべての曲線",
        "crosshair": "十字線を表示",
        "show_removed": "除外した点を表示（赤い ×）",
        "inter_hint": "カーソルを曲線に重ねると座標と行番号が表示されます。行番号は「行番号で除外」にそのまま入力できます。ホイールで拡大縮小、ドラッグで移動、ダブルクリックで元に戻ります。ここは確認用で、出力される図は「静的プレビュー」の内容です。",
        "no_plotly": "plotly がインストールされていません。pip install plotly を実行してください。",
        "show_peaks": "ピーク位置を図に表示（CV）",
        "tbl_only_kept": "除外していない点のみ",
        "tbl_file_sel": "ファイルを選択",
        "tbl_dl": "処理後のデータをダウンロード（CSV）",
        "tbl_hint": "外れ値の除去、平滑化、単位換算をした後のデータです。Origin や Excel にそのまま取り込めます。",
        "warn_range": "除外またはフィット範囲の書式が正しくありません。",
        "preview_caption": "プレビュー（赤い × は除外した点。出力した図には含まれません）",
        "results": "結果",
        "tbl_file": "ファイル",
        "tbl_removed": "除外した点数",
        "tbl_params": "パラメータ",
        "tbl_r2": "R²",
        "fit_fail": "フィット失敗：{err}",
        "sec_export": "画像を出力",
        "fmt": "形式",
        "dpi": "解像度 (dpi)",
        "fname": "ファイル名",
        "dl": "画像をダウンロード",
        "dl_hint": "PDF・SVG はベクター形式で、拡大しても劣化しません。PNG・TIFF はビットマップ形式です。図中の英字は Times New Roman、日本語の漢字・仮名は明朝体で描画します（お使いのパソコンに該当フォントが必要です）。",
    },
}


# ---------------------------------------------------------------- 安全多语言获取函数
def t(key, **kw):
    # 优先找当前语言；找不到则自动退回中文；中文也没有则显示 key 本身，彻底杜绝 KeyError 导致的崩溃
    s = TR.get(lang, {}).get(key) or TR.get("zh", {}).get(key, key)
    return s.format(**kw) if kw else s


# ---------------------------------------------------------------- 数据读取
def _tokens(line):
    return [t for t in re.split(r"[,\t; ]+", line.strip()) if t]


def _is_num_line(line):
    t = _tokens(line)
    if len(t) < 2:
        return False
    try:
        _ = [float(v) for v in t]
        return True
    except ValueError:
        return False


def _has_kana(s):
    return any("\u3040" <= c <= "\u30ff" for c in s)


def decode_bytes(raw: bytes, enc: str, pref: str) -> str:
    if enc != "auto":
        return raw.decode(enc, errors="replace")
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        pass
    gb = sj = None
    try:
        gb = raw.decode("gb18030")
    except UnicodeDecodeError:
        pass
    try:
        sj = raw.decode("cp932")
    except UnicodeDecodeError:
        pass
    if gb is not None and sj is not None:
        if _has_kana(sj):
            return sj
        try:
            gb.encode("gb2312")
            return gb
        except UnicodeEncodeError:
            return sj
    if gb is not None:
        return gb
    if sj is not None:
        return sj
    return raw.decode("latin-1")


def parse_table(raw: bytes, enc: str = "auto", pref: str = "zh") -> pd.DataFrame:
    text = decode_bytes(raw, enc, pref)
    lines = text.splitlines()
    i0 = next((i for i, l in enumerate(lines) if _is_num_line(l)), None)
    if i0 is None:
        raise ValueError("no numeric data")
    n = len(_tokens(lines[i0]))
    rows, i = [], i0
    while i < len(lines) and (not lines[i].strip() or _is_num_line(lines[i])):
        if lines[i].strip():
            t_toks = _tokens(lines[i])
            if len(t_toks) == n:
                rows.append([float(v) for v in t_toks])
        i += 1
    names = [f"col{k}" for k in range(n)]
    for j in range(i0 - 1, -1, -1):
        if lines[j].strip():
            parts = [s.strip() for s in re.split(r"[,\t;]", lines[j]) if s.strip()]
            if len(parts) == n:
                names = parts
            break
    return pd.DataFrame(rows, columns=names)


@st.cache_data
def load_file(name: str, raw: bytes, enc: str, pref: str) -> pd.DataFrame:
    if name.lower().endswith(".xlsx"):
        return pd.read_excel(io.BytesIO(raw)).select_dtypes("number")
    return parse_table(raw, enc, pref)


def make_demo(mode):
    rng = np.random.default_rng(0)
    if mode == "CV":
        e = np.concatenate([np.linspace(-0.2, 0.6, 400), np.linspace(0.6, -0.2, 400)])
        fwd = np.arange(800) < 400
        i = 1e-6 * np.where(fwd, 8 * np.exp(-(((e - 0.32) / 0.07) ** 2)), -6 * np.exp(-(((e - 0.24) / 0.07) ** 2)))
        i = i + 0.6e-6 * e + rng.normal(0, 8e-9, 800)
        i[[120, 530]] += 4e-6
        return pd.DataFrame({"Potential/V": e, "Current/A": i})
    if mode == "CA":
        t = np.linspace(0.05, 60, 600)
        i = 2e-6 / np.sqrt(t) + 1e-7 + rng.normal(0, 2e-8, t.size)
        i[[80, 300]] += 8e-7
        q = np.cumsum(i) * (t[1] - t[0])
        return pd.DataFrame({"Time/sec": t, "Current/A": i, "Charge/C": q})
    t = np.linspace(0, 100, 500)
    e = 0.2 + 0.25 * (1 - np.exp(-t / 20)) + rng.normal(0, 0.002, t.size)
    e[[100, 350]] += 0.05
    return pd.DataFrame({"Time/sec": t, "Potential/V": e})


# ---------------------------------------------------------------- 数据处理
def cscale(p):
    area = p["area"] if p["norm"] else 1.0
    return CUR[p["cur_raw"]] / CUR[p["cur_out"]] / area


def transform(x, y, mode, p):
    if mode == "CV":
        return x + p["shift"], y * cscale(p)
    if mode == "CA":
        return x, y * cscale(p)
    return x, y + p["shift"]


def transform_y2(y2, kind, p):
    if kind == "cur":
        return y2 * cscale(p)
    if kind == "pot":
        return y2 + p["shift"]
    return y2


def transform_x2(x2, kind, p):
    if kind == "pot":
        return x2 + p["shift"]
    if kind == "time":
        return time_conv(x2, p.get("time_unit", "s"))
    return x2


def renumber_x(x, keep):
    # 剔除点后，让保留点的坐标依次递补：第一个保留点归零，连续点保持原步长，断开处用中位步长衔接
    x = x.copy()
    kept = np.where(keep)[0]
    if len(kept) == 0:
        return x
    diffs = np.diff(x)
    pos = diffs[np.abs(diffs) > 1e-12]
    dx_default = np.median(pos) if len(pos) > 0 else 1.0
    new = np.zeros(len(kept))
    for i in range(1, len(kept)):
        a, b = kept[i - 1], kept[i]
        step = x[b] - x[a] if b == a + 1 else dx_default
        new[i] = new[i - 1] + step
    x[keep] = new
    return x


def shift_head(x, keep):
    # 只收拢开头：第一个保留点归零，中间被剔除的空档保持原样
    x = x.copy()
    first = np.where(keep)[0][0]
    return x - x[first]


def time_conv(x, unit):
    # 原始时间单位是秒
    if unit == "ms":
        return x * 1000.0
    if unit == "min":
        return x / 60.0
    if unit == "h":
        return x / 3600.0
    return x


def hampel(y, window=7, n_sigma=3.0):
    s = pd.Series(y)
    med = s.rolling(window, center=True, min_periods=1).median()
    mad = (s - med).abs().rolling(window, center=True, min_periods=1).median()
    thr = (n_sigma * 1.4826 * mad).replace(0, np.nan).fillna(np.inf)
    return ((s - med).abs() > thr).to_numpy()


def smooth_1d(y, method, win, poly, sigma):
    n = len(y)
    if method == "none" or n < 5:
        return y
    if method == "savgol":
        w = win if win % 2 else win + 1
        w = min(w, n if n % 2 else n - 1)
        order = min(poly, w - 1)
        if w <= order:
            return y
        return savgol_filter(y, w, order)
    if method == "ma":
        return pd.Series(y).rolling(win, center=True, min_periods=1).mean().to_numpy()
    return gaussian_filter1d(y, sigma, mode="nearest")


def parse_idx(s):
    out = set()
    for tok in re.split(r"[,，\s]+", s.strip()):
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(tok))
    return out


def parse_ranges(s):
    out = []
    for tok in re.split(r"[;；]", s):
        if ":" in tok:
            a, b = tok.split(":", 1)
            lo, hi = float(a), float(b)
            out.append((min(lo, hi), max(lo, hi)))
    return out


def parse_float(s):
    s = s.strip()
    return float(s) if s else None


def _sig(v, n=4):
    v = float(v)
    return float(f"{v:.{n}g}") if np.isfinite(v) else None


def cv_summary(x, y, keep):
    xs, ys = x[keep], y[keep]
    n = len(xs)
    if n < 30:
        return None
    sgn = np.sign(np.diff(xs))
    for i in range(len(sgn)):
        if sgn[i] == 0:
            sgn[i] = sgn[i - 1] if i else 1
    direction = np.append(sgn, sgn[-1])
    cuts = np.flatnonzero(np.diff(direction) != 0) + 1
    segs = [sg for sg in np.split(np.arange(n), cuts) if len(sg) >= max(10, int(0.05 * n))]
    span = float(np.ptp(ys)) or 1.0
    ana, cat = [], []
    for sg in segs:
        yy = ys[sg]
        sign = 1 if direction[sg[0]] > 0 else -1
        pk, props = find_peaks(sign * yy, prominence=0.1 * span)
        if len(pk):
            j = pk[np.argmax(props["prominences"])]
            (ana if sign > 0 else cat).append((float(xs[sg][j]), float(yy[j])))
    out = {"n_sweeps_detected": len(segs)}
    if ana:
        out["Epa"], out["ipa"] = _sig(ana[0][0]), _sig(ana[0][1])
    if cat:
        out["Epc"], out["ipc"] = _sig(cat[0][0]), _sig(cat[0][1])
    if ana and cat:
        out["dEp_mV"] = _sig(1000 * (ana[0][0] - cat[0][0]))
        out["E_half"] = _sig((ana[0][0] + cat[0][0]) / 2)
        if cat[0][1] != 0:
            out["ipa_over_abs_ipc"] = _sig(ana[0][1] / abs(cat[0][1]))
    return out


def trace_summary(x, y, keep):
    xs, ys = x[keep], y[keep]
    n = len(xs)
    if n < 10:
        return None
    k = max(3, n // 10)
    d = np.diff(ys)
    noise = 1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2)
    half = slice(n // 2, None)
    slope = np.polyfit(xs[half], ys[half], 1)[0] if np.ptp(xs[half]) > 0 else np.nan
    return {
        "y_mean_first_10pct": _sig(ys[:k].mean()),
        "y_mean_last_10pct": _sig(ys[-k:].mean()),
        "y_min": _sig(ys.min()),
        "y_max": _sig(ys.max()),
        "noise_sd_estimate": _sig(noise),
        "slope_of_last_half_per_x_unit": _sig(slope),
    }


def downsample(x, y, keep, n=60):
    xs, ys = x[keep], y[keep]
    idx = np.linspace(0, len(xs) - 1, min(n, len(xs))).astype(int)
    return [[_sig(a), _sig(b)] for a, b in zip(xs[idx], idx)]


def get_api_key(typed):
    if typed.strip():
        return typed.strip()
    try:
        k = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        k = ""
    return k or os.environ.get("GEMINI_API_KEY", "")


AI_SYSTEM = (
    "You are an assistant for electrochemistry data analysis. You receive a JSON summary of already-processed data "
    "(cyclic voltammetry, chronoamperometry or chronopotentiometry) produced by a plotting tool; the numbers were "
    "computed by the tool, not by you. Base every statement on the provided numbers and never invent values. "
    "Separate clearly: (1) what the numbers show, (2) possible interpretations and the assumptions they depend on, "
    "(3) what to check or do next. Point out limits of this analysis (for example: peak currents are not "
    "baseline-corrected, sweep direction is inferred from the potential, smoothing can distort peaks, few points, "
    "unknown scan rate or electrode area). Use standard criteria such as the reversibility criteria or the "
    "Randles-Sevcik / Cottrell relations only when the data supports them, and say when information is insufficient "
    "for a conclusion. Keep it concise (about 400 words at most), with short paragraphs or bullet lists. "
    "This is for research use. Reply in {language}."
)
AI_LANG = {"zh": "Simplified Chinese", "ja": "Japanese"}


def call_gemini(api_key, model, system, user_text, max_tokens=1500):
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=model,
        contents=user_text,
        config=types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
        ),
    )

    return response.text or ""


# ---------------------------------------------------------------- 拟合
MODELS = {
    "linear": lambda x, y: (lambda x, a, b: a * x + b, [1.0, 0.0], ["a", "b"]),
    "exp": lambda x, y: (
        lambda x, a, tau, c: a * np.exp(-x / tau) + c,
        [y[0] - y[-1], ((x[-1] - x[0]) / 3) or 1.0, y[-1]],
        ["a", "τ", "c"],
    ),
    "cottrell": lambda x, y: (
        lambda x, a, c: a / np.sqrt(x) + c,
        [y[0] * np.sqrt(x[0]), 0.0],
        ["a", "c"],
    ),
}


def do_fit(x, y, kind, deg):
    if kind == "poly":
        coef, cov = np.polyfit(x, y, deg, cov=True)
        vals, err = coef, np.sqrt(np.diag(cov))
        names = [f"c{deg - i}" for i in range(deg + 1)]
        func = lambda xx: np.polyval(coef, xx)
    else:
        f, p0, names = MODELS[kind](x, y)
        popt, pcov = curve_fit(f, x, y, p0=p0, maxfev=20000)
        vals, err = popt, np.sqrt(np.diag(pcov))
        func = lambda xx: f(xx, *popt)
    res = y - func(x)
    r2 = 1 - np.sum(res**2) / np.sum((y - y.mean()) ** 2)
    return func, names, vals, err, r2


# ---------------------------------------------------------------- 绘图样式
def set_style(p):
    fs = p["fs"]
    plt.rcParams.update(
        {
            "font.family": FONTS,
            "mathtext.fontset": "stix",
            "mathtext.default": "regular",
            "font.size": fs,
            "axes.labelsize": fs + 1,
            "legend.fontsize": max(fs - 1, 5),
            "axes.linewidth": 0.8,
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "xtick.minor.size": 2,
            "ytick.minor.size": 2,
            "lines.linewidth": 1.2,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


_SUP = str.maketrans(
    {
        **dict(zip("0123456789+-=()", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾")),
        **dict(zip("abcdefghijklmnoprstuvwxyz", "ᵃᵇᶜᵈᵉᶠᵍʰⁱʲᵏˡᵐⁿᵒᵖʳˢᵗᵘᵛʷˣʸᶻ")),
        **dict(zip("ABDEGHIJKLMNOPRTUVW", "ᴬᴮᴰᴱᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾᴿᵀᵁⱽᵂ")),
    }
)
_SUB = str.maketrans(
    {
        **dict(zip("0123456789+-=()", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎")),
        **dict(zip("aehijklmnoprstuvx", "ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓ")),
    }
)
_GREEK = {
    r"\alpha": "α", r"\beta": "β", r"\gamma": "γ", r"\delta": "δ", r"\Delta": "Δ", r"\epsilon": "ε",
    r"\theta": "θ", r"\lambda": "λ", r"\mu": "µ", r"\pi": "π", r"\rho": "ρ", r"\sigma": "σ",
    r"\tau": "τ", r"\phi": "φ", r"\omega": "ω", r"\Omega": "Ω", r"\times": "×", r"\cdot": "·",
    r"\pm": "±", r"\approx": "≈",
}


def _uni_math(m):
    body = m.group(1).replace("\\mathrm", "").replace("\\,", " ")
    body = body.replace("^\\circ", "°").replace("^{\\circ}", "°")
    for k in sorted(_GREEK, key=len, reverse=True):
        body = re.sub(re.escape(k) + r"(?![A-Za-z]) ?", lambda _m, v=_GREEK[k]: v, body)
    body = re.sub(r"\^\{([^}]*)\}|\^(\S)", lambda g: (g.group(1) or g.group(2)).translate(_SUP), body)
    body = re.sub(r"_\{([^}]*)\}|_(\S)", lambda g: (g.group(1) or g.group(2)).translate(_SUB), body)
    return body.replace("{", "").replace("}", "")


def cjk_safe(label):
    if "$" in label and any(ord(c) >= 0x2E80 for c in label):
        return re.sub(r"\$([^$]*)\$", _uni_math, label)
    return label


def pretty(colname):
    s = re.sub(r"/\s*sec\b", "/s", str(colname))
    return re.sub(r"\s*/\s*", " / ", s).strip()


def short(colname):
    return str(colname).split("/")[0].strip()


def cur_label(p):
    name = "Current density" if p["norm"] else "Current"
    unit = f"{p['cur_out']} cm$^{{-2}}$" if p["norm"] else p["cur_out"]
    return f"{name} / {unit}"


def pot_label(p):
    if p["ref"] == "none":
        return "Potential / V"
    ref = "RHE" if p["to_rhe"] else p["ref"]
    return f"Potential / V vs. {ref}"


def axis_labels(mode, p):
    x_unit = p.get("time_unit", "s")
    if mode == "CV":
        return pot_label(p), cur_label(p)
    if mode == "CA":
        return f"Time / {x_unit}", cur_label(p)
    return f"Time / {x_unit}", pot_label(p)


def y2_label(p, y2kind, y2name):
    if y2kind == "cur":
        return cur_label(p)
    if y2kind == "pot":
        return pot_label(p)
    return pretty(y2name)


def x2_label(p, x2kind, x2name):
    if x2kind == "time":
        return f"Time / {p.get('time_unit', 's')}"
    if x2kind == "pot":
        return pot_label(p)
    return pretty(x2name)


def apply_ticks(ax, ax2, ax3, p):
    tm, tn, d = p["tick_marks"], p["tick_nums"], p["tick_dir"]
    ax.tick_params(axis="both", which="both", direction=d)
    top_marks = tm["top"] if not ax3 else False
    top_nums = tn["top"] if not ax3 else False
    
    ax.tick_params(axis="x", which="both", bottom=tm["bottom"], top=top_marks, labelbottom=tn["bottom"], labeltop=top_nums)
    ax.tick_params(axis="y", which="both", left=tm["left"], labelleft=tn["left"])
    if ax2 is None:
        ax.tick_params(axis="y", which="both", right=tm["right"], labelright=tn["right"])
    else:
        ax.tick_params(axis="y", which="both", right=False, labelright=False)
        ax2.tick_params(axis="y", which="both", direction=d, left=False, labelleft=False, right=tm["right"], labelright=tn["right"])
    
    if ax3 is not None:
        ax3.tick_params(axis="x", which="both", direction=d, top=tm["top"], labeltop=tn["top"], bottom=False, labelbottom=False)

    if p["minor"]:
        ax.minorticks_on()
        if ax2 is not None:
            ax2.minorticks_on()
        if ax3 is not None:
            ax3.minorticks_on()
    else:
        ax.minorticks_off()
        if ax2 is not None:
            ax2.minorticks_off()
        if ax3 is not None:
            ax3.minorticks_off()


def color_xaxis(a, side, c):
    if a.xaxis:
        a.xaxis.label.set_color(c)
        a.tick_params(axis="x", which="both", colors=c)
    a.spines[side].set_color(c)


def color_yaxis(a, side, c):
    if a.yaxis:
        a.yaxis.label.set_color(c)
        a.tick_params(axis="y", which="both", colors=c)
    a.spines[side].set_color(c)


def build_fig(results, mode, p, show_removed):
    set_style(p)
    fig, ax = plt.subplots(figsize=(p["w"] * MM, p["h"] * MM), constrained_layout=True)
    ax2 = ax.twinx() if p["dual"] else None
    ax3 = ax.twiny() if p.get("dual_x") else None
    
    handles, labels = [], []
    sm_left = p["sm_method"] != "none"
    sm_right = sm_left and p["sm_right"]

    def draw(a, x, ys, yr, keep, color, label, smoothed, mark_removed=True):
        if smoothed and p["show_raw"]:
            if p["scatter"]:
                a.plot(x[keep], yr[keep], "o", ms=1.6, color=color, alpha=0.3, ls="none")
            else:
                a.plot(x[keep], yr[keep], color=color, alpha=0.3, lw=0.8)
        if smoothed or not p["scatter"]:
            (h,) = a.plot(x[keep], ys[keep], color=color, label=label)
        else:
            (h,) = a.plot(x[keep], ys[keep], "o", ms=2, color=color, label=label)
        if show_removed and mark_removed and (~keep).any():
            a.plot(x[~keep], yr[~keep], "x", color="red", ms=4, mew=0.8)
        handles.append(h)
        labels.append(label)

    xl, yl = axis_labels(mode, p)
    actual_yl = p["lab_y"] or yl
    actual_y2l = (p["lab_y2"] or y2_label(p, p["y2kind"], results[0]["y2name"])) if p["dual"] else ""

    actual_x2l = (p["lab_x2"] or x2_label(p, p["x2kind"], results[0]["x2name"])) if ax3 is not None else ""

    multi = len(results) > 1
    custom_legs = p.get("custom_legends", [])
    custom_x2 = p.get("custom_legends_x2", [])
    for k, r in enumerate(results):
        cl, cr = p["colors"][k]
        c_left, c_right = custom_legs[k] if k < len(custom_legs) else ("", "")
        c_x2 = custom_x2[k] if k < len(custom_x2) else ""
        if p["dual"]:
            default_left = f"{r['name']} ({short(actual_yl)})" if multi else actual_yl
            default_right = f"{r['name']} ({short(actual_y2l)})" if multi else actual_y2l
            left_name = c_left if c_left else default_left
            right_name = c_right if c_right else default_right
        else:
            left_name = c_left if c_left else r['name']

        draw(ax, r["x"], r["y"], r["y_raw"], r["keep"], cl, left_name, sm_left)
        if p["dual"]:
            draw(ax2, r["x"], r["y2"], r["y2_raw"], r["keep"], cr, right_name, sm_right)
        if ax3 is not None:
            top_name = c_x2 if c_x2 else f"{r['name']} ({short(actual_x2l)})"
            draw(ax3, r["x2"], r["y"], r["y_raw"], r["keep"], p["colors_x2"][k], top_name, sm_left, mark_removed=False)
        if r["fit"] is not None:
            xx, yy, r2 = r["fit"]
            (h,) = ax.plot(xx, yy, "--", color=p["fit_color"], lw=1.0)
            handles.append(h)
            labels.append(f"Fit ($R^2$ = {r2:.4f})")

    if p["show_peaks"] and mode == "CV":
        for k, r in enumerate(results):
            pk = r.get("peaks") or {}
            cl = p["colors"][k][0]
            if "Epa" in pk:
                ax.plot(pk["Epa"], pk["ipa"], "v", color=cl, ms=4)
                ax.annotate(f"$E_{{pa}}$ = {pk['Epa']:.3f} V", (pk["Epa"], pk["ipa"]), textcoords="offset points",
                            xytext=(6, 6), fontsize=max(p["fs"] - 2, 5), color=cl)
            if "Epc" in pk:
                ax.plot(pk["Epc"], pk["ipc"], "^", color=cl, ms=4)
                ax.annotate(f"$E_{{pc}}$ = {pk['Epc']:.3f} V", (pk["Epc"], pk["ipc"]), textcoords="offset points",
                            xytext=(6, -12), fontsize=max(p["fs"] - 2, 5), color=cl)

    xl, yl = axis_labels(mode, p)
    ax.set_xlabel(cjk_safe(p["lab_x"] or xl))
    ax.set_ylabel(cjk_safe(p["lab_y"] or yl))

    if ax3 is not None:
        ax3.set_xlabel(cjk_safe(actual_x2l))
        ax.spines["top"].set_visible(False)
        ax3.spines[["left", "right", "bottom"]].set_visible(False)

    if p["dual"]:
        ax.spines["right"].set_visible(False)
        ax2.spines[["left", "top", "bottom"]].set_visible(False)
        ax2.set_ylabel(cjk_safe(p["lab_y2"] or y2_label(p, p["y2kind"], results[0]["y2name"])))
        
    if p["xlim"] != (None, None):
        ax.set_xlim(*p["xlim"])
    if p["ylim"] != (None, None):
        ax.set_ylim(*p["ylim"])
    if p["dual"] and p["y2lim"] != (None, None):
        ax2.set_ylim(*p["y2lim"])
    if ax3 is not None and p.get("x2lim") and p["x2lim"] != (None, None):
        ax3.set_xlim(*p["x2lim"])

    skipped = []

    def set_step(a, axis, step):
        if step is None or not a:
            return
        lo, hi = a.get_xlim() if axis == "x" else a.get_ylim()
        if abs(hi - lo) / step > 200:
            skipped.append(axis)
            return
        (a.xaxis if axis == "x" else a.yaxis).set_major_locator(MultipleLocator(step))

    set_step(ax, "x", p["xstep"])
    set_step(ax, "y", p["ystep"])
    if p["dual"]:
        set_step(ax2, "y", p["y2step"])
    if ax3:
        set_step(ax3, "x", p.get("x2step"))

    apply_ticks(ax, ax2, ax3, p)
    color_xaxis(ax, "bottom", p["xb_col"])
    color_yaxis(ax, "left", p["yl_col"])
    if p["dual"]:
        color_yaxis(ax2, "right", p["yr_col"])
    if ax3 is not None:
        color_xaxis(ax3, "top", p["xt_col"])
    fig._tick_skipped = bool(skipped)

    if p["legend"] and handles:
        kw = dict(ncol=p.get("legend_ncol", 1), frameon=p.get("legend_frame", False))
        if p.get("legend_loc") == "manual":
            kw.update(loc="upper left", bbox_to_anchor=p["legend_xy"])
        else:
            kw.update(loc=p.get("legend_loc", "best"))
        if kw["frameon"]:
            kw.update(edgecolor="black", fancybox=False, framealpha=1.0)
        leg = (ax3 or ax2 or ax).legend(handles, labels, **kw)
        if kw["frameon"]:
            leg.get_frame().set_linewidth(0.8)
    return fig


# ---------------------------------------------------------------- Plotly 交互图表
PLOTLY_FONT = "Times New Roman, Yu Mincho, MS Mincho, Hiragino Mincho ProN, serif"


def html_label(s):
    def conv(m):
        body = m.group(1).replace("\\mathrm", "").replace("\\,", " ")
        for k in sorted(_GREEK, key=len, reverse=True):
            body = re.sub(re.escape(k) + r"(?![A-Za-z]) ?", lambda _m, v=_GREEK[k]: v, body)
        body = re.sub(r"\^\{([^}]*)\}|\^(\S)", lambda g: f"<sup>{g.group(1) or g.group(2)}</sup>", body)
        body = re.sub(r"_\{([^}]*)\}|_(\S)", lambda g: f"<sub>{g.group(1) or g.group(2)}</sub>", body)
        return body.replace("{", "").replace("}", "")

    return re.sub(r"\$([^$]*)\$", conv, s)


def plain_label(s):
    return re.sub(r"\$([^$]*)\$", _uni_math, s)


def _axis_range(lim, arrays):
    lo, hi = lim
    if lo is None and hi is None:
        return None
    vals = np.concatenate(arrays) if arrays else np.array([0.0, 1.0])
    d_lo, d_hi = float(np.nanmin(vals)), float(np.nanmax(vals))
    pad = 0.03 * ((d_hi - d_lo) or 1.0)
    return [lo if lo is not None else d_lo - pad, hi if hi is not None else d_hi + pad]


def build_plotly(results, mode, p, hover_mode, spikes, show_removed):
    xl, yl = axis_labels(mode, p)
    xt, yt = html_label(p["lab_x"] or xl), html_label(p["lab_y"] or yl)
    y2t = html_label(p["lab_y2"] or y2_label(p, p["y2kind"], results[0]["y2name"])) if p["dual"] else ""
    actual_x2l = (p["lab_x2"] or x2_label(p, p["x2kind"], results[0]["x2name"])) if p["dual_x"] else ""
    x2t = html_label(actual_x2l)
    sm_left = p["sm_method"] != "none"
    sm_right = sm_left and p["sm_right"]
    multi = len(results) > 1
    fig = go.Figure()
    all_x, all_x2, all_y, all_y2 = [], [], [], []

    def tip(name, ytitle, xtitle=None):
        return f"<b>{name}</b><br>{xtitle or xt}: %{{x:.5g}}<br>{ytitle}: %{{y:.5g}}<br>row %{{customdata}}<extra></extra>"

    def add(r, y_s, y_r, color, name, ytitle, axis, smoothed, xs=None, xaxis="x", xtitle=None, mark_removed=True):
        xs = r["x"] if xs is None else xs
        keep = r["keep"]
        if smoothed and p["show_raw"]:
            fig.add_trace(
                go.Scatter(
                    x=xs[keep], y=y_r[keep], mode="markers" if p["scatter"] else "lines", opacity=0.3,
                    line=dict(color=color, width=1), marker=dict(color=color, size=3),
                    hoverinfo="skip", showlegend=False, yaxis=axis, xaxis=xaxis,
                )
            )
        as_lines = smoothed or not p["scatter"]
        fig.add_trace(
            go.Scatter(
                x=xs[keep], y=y_s[keep], mode="lines" if as_lines else "markers", name=name,
                line=dict(color=color, width=1.8), marker=dict(color=color, size=5),
                customdata=r["idx"][keep], hovertemplate=tip(name, ytitle, xtitle), yaxis=axis, xaxis=xaxis,
            )
        )
        if show_removed and mark_removed and (~keep).any():
            fig.add_trace(
                go.Scatter(
                    x=xs[~keep], y=y_r[~keep], mode="markers", name="removed", showlegend=False,
                    marker=dict(color="red", symbol="x", size=7), customdata=r["idx"][~keep],
                    hovertemplate=tip("removed", ytitle, xtitle), yaxis=axis, xaxis=xaxis,
                )
            )
        (all_x2 if xaxis == "x2" else all_x).append(xs[keep])
        (all_y if axis == "y" else all_y2).append(y_s[keep])

    actual_yl = p["lab_y"] or yl
    actual_y2l = (p["lab_y2"] or y2_label(p, p["y2kind"], results[0]["y2name"])) if p["dual"] else ""

    custom_legs = p.get("custom_legends", [])
    custom_x2 = p.get("custom_legends_x2", [])
    for k, r in enumerate(results):
        cl, cr = p["colors"][k]
        c_left, c_right = custom_legs[k] if k < len(custom_legs) else ("", "")
        c_x2 = custom_x2[k] if k < len(custom_x2) else ""
        if p["dual"]:
            default_left = f"{r['name']} ({short(actual_yl)})" if multi else actual_yl
            default_right = f"{r['name']} ({short(actual_y2l)})" if multi else actual_y2l
            left_name = c_left if c_left else default_left
            right_name = c_right if c_right else default_right
        else:
            left_name = c_left if c_left else r['name']

        add(r, r["y"], r["y_raw"], cl, left_name, yt, "y", sm_left)
        if p["dual"]:
            add(r, r["y2"], r["y2_raw"], cr, right_name, y2t, "y2", sm_right)
        if p["dual_x"]:
            top_name = c_x2 if c_x2 else f"{r['name']} ({short(actual_x2l)})"
            add(
                r, r["y"], r["y_raw"], p["colors_x2"][k], top_name, yt, "y", sm_left,
                xs=r["x2"], xaxis="x2", xtitle=x2t, mark_removed=False,
            )
        if p["show_peaks"] and mode == "CV":
            pk = r.get("peaks") or {}
            for key, sym, pos in (("a", "triangle-down", "top center"), ("c", "triangle-up", "bottom center")):
                if f"E p{key}".replace(" ", "") in pk:
                    ex, iy = pk[f"Ep{key}"], pk[f"ip{key}"]
                    fig.add_trace(
                        go.Scatter(
                            x=[ex], y=[iy], mode="markers+text", text=[f"E<sub>p{key}</sub> = {ex:.3f} V"], textposition=pos,
                            marker=dict(color=cl, symbol=sym, size=9), showlegend=False,
                            hovertemplate=f"<b>E<sub>p{key}</sub></b> = {ex:.4g}<br>i<sub>p{key}</sub> = {iy:.4g}<extra></extra>",
                        )
                    )
        if r["fit"] is not None:
            xx, yy, r2 = r["fit"]
            fig.add_trace(
                go.Scatter(
                    x=xx, y=yy, mode="lines", name=f"Fit (R² = {r2:.4f})", line=dict(color=p["fit_color"], dash="dash", width=1.4),
                    hovertemplate=f"<b>Fit</b><br>{xt}: %{{x:.5g}}<br>{yt}: %{{y:.5g}}<extra></extra>",
                )
            )

    ticks = "inside" if p["tick_dir"] == "in" else "outside"
    spike = dict(showspikes=spikes, spikemode="across", spikesnap="data", spikethickness=1, spikedash="dot", spikecolor="#666")
    fig.update_layout(
        template="simple_white", height=470,
        margin=dict(l=70, r=70 if p["dual"] else 20, t=60 if p["dual_x"] else 20, b=60),
        font=dict(family=PLOTLY_FONT, size=13), hovermode="closest" if hover_mode == "closest" else "x unified",
        showlegend=bool(multi or p["dual"] or p["dual_x"] or any(r["fit"] for r in results)),
        legend=dict(orientation="h", y=1.22 if p["dual_x"] else 1.1),
        xaxis=dict(
            title=dict(text=xt, font=dict(color=p["xb_col"])), tickfont=dict(color=p["xb_col"]), linecolor=p["xb_col"],
            showline=True, mirror=not p["dual_x"], ticks=ticks, range=_axis_range(p["xlim"], all_x), dtick=p["xstep"], **spike,
        ),
        yaxis=dict(
            title=dict(text=yt, font=dict(color=p["yl_col"])), tickfont=dict(color=p["yl_col"]), linecolor=p["yl_col"],
            showline=True, mirror=not p["dual"], ticks=ticks, range=_axis_range(p["ylim"], all_y), dtick=p["ystep"], **spike,
        ),
    )
    if p["dual"]:
        fig.update_layout(
            yaxis2=dict(
                overlaying="y", side="right", showgrid=False, showline=True, ticks=ticks,
                title=dict(text=y2t, font=dict(color=p["yr_col"])), tickfont=dict(color=p["yr_col"]), linecolor=p["yr_col"],
                range=_axis_range(p["y2lim"], all_y2), dtick=p["y2step"],
            )
        )
    if p["dual_x"]:
        fig.update_layout(
            xaxis2=dict(
                overlaying="x", side="top", showgrid=False, showline=True, ticks=ticks,
                title=dict(text=x2t, font=dict(color=p["xt_col"])),
                tickfont=dict(color=p["xt_col"]),
                linecolor=p["xt_col"],
                range=_axis_range(p.get("x2lim"), all_x2), dtick=p.get("x2step"),
                **spike,
            )
        )
    return fig


def processed_frame(r, mode, p, only_kept):
    xl, yl = axis_labels(mode, p)
    xn, yn = plain_label(p["lab_x"] or xl), plain_label(p["lab_y"] or yl)
    data = {"row": r["idx"], xn: r["x"], yn: r["y"]}
    if p["sm_method"] != "none":
        data[yn + " (raw)"] = r["y_raw"]
    if p["dual"]:
        y2n = plain_label(p["lab_y2"] or y2_label(p, p["y2kind"], r["y2name"]))
        data[y2n if y2n not in data else y2n + " (right)"] = r["y2"]
    if p["dual_x"]:
        x2n = plain_label(p["lab_x2"] or x2_label(p, p["x2kind"], r["x2name"]))
        data[x2n if x2n not in data else x2n + " (top)"] = r["x2"]
    df = pd.DataFrame(data)
    df["kept"] = r["keep"]
    if only_kept:
        df = df[df["kept"]].drop(columns="kept")
    return df.reset_index(drop=True)


def export_bytes(fig, fmt, dpi):
    buf = io.BytesIO()
    kw = {"pil_kwargs": {"compression": "tiff_lzw"}} if fmt == "tiff" else {}
    text = "".join(t.get_text() for t in fig.findobj(matplotlib.text.Text))
    cjk = any(ord(c) >= 0x2E80 for c in text)
    with plt.rc_context({"svg.fonttype": "path" if cjk else "none"}):
        fig.savefig(buf, format=fmt, dpi=dpi, **kw)
    return buf.getvalue()


# ---------------------------------------------------------------- 界面 / UI
st.set_page_config(page_title="Echem Plotter", layout="wide")
sb = st.sidebar
lang = sb.radio("语言 / 言語", ["zh", "ja"], format_func={"zh": "中文", "ja": "日本語"}.get, horizontal=True, key="lang")

st.title(t("title"))
mode = sb.radio(t("mode_label"), ["CV", "CA", "CP"], format_func=lambda m: t("mode_" + m), key="mode")
use_demo = sb.checkbox(t("use_demo"), value=True, key="demo")
ENC = {"auto": "auto", "utf8": "utf-8-sig", "gbk": "gb18030", "sjis": "cp932"}
enc_key = sb.selectbox(t("enc"), list(ENC), format_func=lambda k: t("enc_" + k), key="enc")

files = st.file_uploader(t("upload"), type=["txt", "csv", "dat", "xlsx"], accept_multiple_files=True, key="files")

dfs, demo_shown = {}, False
for f in files:
    try:
        dfs[f.name] = load_file(f.name, f.getvalue(), ENC[enc_key], lang)
    except Exception as e:
        st.error(t("parse_fail", name=f.name, err=e))
if not dfs:
    if use_demo:
        dfs = {f"demo_{mode}": make_demo(mode)}
        demo_shown = True
    else:
        st.info(t("info_upload"))
        st.stop()
if demo_shown:
    st.caption(t("demo_note"))

cols = list(next(iter(dfs.values())).columns)
if len(cols) < 2:
    st.error(t("err_two_cols"))
    st.stop()
sig = f"{mode}_{len(cols)}"

p = {"cur_raw": "A", "cur_out": "µA", "norm": False, "area": 1.0, "ref": list(REF)[0], "to_rhe": False, "shift": 0.0}

AXC = ["black", "follow", "custom"]


def _h(key):
    st.markdown(f"**{t(key)}**")


# ================================================================ 1. 数据
with sb.expander(t("grp_data"), expanded=True):
    _h("sub_cols")
    xi = st.selectbox(t("xcol"), range(len(cols)), index=0, format_func=lambda i: cols[i], key=f"xi_{sig}")
    yi = st.selectbox(t("ycol"), range(len(cols)), index=1, format_func=lambda i: cols[i], key=f"yi_{sig}")
    p["dual"] = st.checkbox(t("dual"), key="dual")
    p["dual_x"] = st.checkbox(t("dual_x"), key="dual_x")

    y2i, y2kind = 0, "other"
    if p["dual"]:
        y2i = st.selectbox(t("y2col"), range(len(cols)), index=min(2, len(cols) - 1), format_func=lambda i: cols[i], key=f"y2i_{sig}")
        nm = str(cols[y2i]).lower()
        guess = 0 if "current" in nm else (1 if "potential" in nm else 2)
        y2kind = st.selectbox(t("y2kind"), ["cur", "pot", "other"], index=guess, format_func=lambda k: t("kind_" + k), key=f"y2kind_{y2i}")
    p["y2kind"] = y2kind

    x2i, x2kind = 0, "other"
    if p["dual_x"]:
        t_idx = next((i for i, c in enumerate(cols) if "time" in str(c).lower() and i != xi), None)
        x2_default = t_idx if t_idx is not None else min(2, len(cols) - 1)
        x2i = st.selectbox(t("x2col"), range(len(cols)), index=x2_default, format_func=lambda i: cols[i], key=f"x2i_{sig}")
        nm = str(cols[x2i]).lower()
        guess = 0 if "time" in nm else (1 if "potential" in nm else 2)
        x2kind = st.selectbox(t("x2kind"), ["time", "pot", "other"], index=guess, format_func=lambda k: t("kind_" + k), key=f"x2kind_{x2i}")
        st.caption(t("x2_note"))
    p["x2kind"] = x2kind

    st.divider()
    _h("sub_rows")
    c1, c2 = st.columns(2)
    p_start = c1.number_input(t("start"), min_value=0, value=0, step=1, key="start")
    p_end = c2.number_input(t("end"), min_value=0, value=0, step=1, key="end")

    need_cur = mode in ("CV", "CA") or (p["dual"] and y2kind == "cur")
    need_pot = mode in ("CV", "CP") or (p["dual"] and y2kind == "pot") or (p["dual_x"] and x2kind == "pot")
    need_time = mode in ("CA", "CP") or (p["dual_x"] and x2kind == "time")
    if need_cur or need_pot or need_time:
        st.divider()
        _h("sec_units")
    p["time_unit"] = st.selectbox(t("time_unit"), ["s", "ms", "min", "h"], index=0, key="time_unit") if need_time else "s"
    if need_cur:
        p["cur_raw"] = st.selectbox(t("cur_raw"), list(CUR), index=0, key="cur_raw")
        p["cur_out"] = st.selectbox(t("cur_out"), list(CUR), index=2, key="cur_out")
        p["norm"] = st.checkbox(t("norm"), key="norm")
        if p["norm"]:
            p["area"] = st.number_input(t("area"), min_value=1e-6, value=0.07, format="%.4f", key="area")
    if need_pot:
        ref_sel = st.selectbox(
            t("ref"), ["none"] + list(REF) + ["custom"], index=1,
            format_func=lambda r: t("ref_none") if r == "none" else (t("ref_custom") if r == "custom" else r), key="ref",
        )
        p["ref"], p["ref_E"] = ref_sel, REF.get(ref_sel, 0.0)
        if ref_sel == "custom":
            p["ref"] = st.text_input(t("ref_custom_name"), key="ref_custom_name").strip() or "custom"
        if ref_sel != "none":
            p["to_rhe"] = st.checkbox(t("to_rhe"), key="to_rhe")
        if p["to_rhe"]:
            if ref_sel == "custom":
                p["ref_E"] = st.number_input(t("ref_custom_E"), value=0.0, step=0.001, format="%.3f", key="ref_custom_E")
            ph = st.number_input(t("ph"), value=7.0, step=0.1, key="ph")
            p["shift"] = p["ref_E"] + 0.0591 * ph

# ================================================================ 2. 数据处理
with sb.expander(t("grp_proc")):
    _h("sec_clean")
    use_hampel = st.checkbox(t("hampel"), key="hampel")
    win = st.slider(t("win"), 3, 51, 11, step=2, key="win") if use_hampel else 11
    nsig = st.slider(t("nsig"), 1.0, 8.0, 5.0, 0.5, key="nsig") if use_hampel else 5.0
    if use_hampel:
        st.caption(t("hampel_note"))
    ex_range = st.text_input(t("ex_range"), key="ex_range")
    ex_idx = st.text_input(t("ex_idx"), key="ex_idx")
    p["reset_mode"] = st.selectbox(t("reset_mode"), ["all", "head", "off"], index=0, format_func=lambda k: t("reset_" + k), key="reset_mode")
    st.caption(t("reset_note"))

    st.divider()
    _h("sec_smooth")
    p["sm_method"] = st.selectbox(t("sm_method"), ["none", "savgol", "ma", "gauss"], format_func=lambda k: t("sm_" + k), key="sm_method")
    sm_win, sm_poly, sm_sigma = 11, 3, 2.0
    p["sm_right"], p["show_raw"] = False, False
    if p["sm_method"] != "none":
        if p["sm_method"] in ("savgol", "ma"):
            sm_win = st.slider(t("sm_win"), 3, 201, 11, step=2, key="sm_win")
        if p["sm_method"] == "savgol":
            sm_poly = st.slider(t("sm_poly"), 1, 6, 3, key="sm_poly")
        if p["sm_method"] == "gauss":
            sm_sigma = st.slider(t("sm_sigma"), 0.5, 30.0, 2.0, 0.5, key="sm_sigma")
        if p["dual"]:
            p["sm_right"] = st.checkbox(t("sm_right"), key="sm_right")
        p["show_raw"] = st.checkbox(t("sm_raw"), key="sm_raw")
        st.caption(t("sm_note"))

    st.divider()
    _h("sec_fit")
    kind = st.selectbox(t("fit_model"), FITS[mode], format_func=lambda k: t("fit_" + k), key=f"kind_{mode}")
    deg = st.slider(t("deg"), 1, 6, 2, key="deg") if kind == "poly" else 2
    fit_range = st.text_input(t("fit_range"), key="fit_range") if kind != "none" else ""

# ================================================================ 3. 坐标轴
# 每个轴一个小节，顺序统一：标签文字 → 范围 → 主刻度间隔 → 轴的颜色；没开的轴不显示
with sb.expander(t("grp_axes")):
    # ---- 下方 x 轴
    _h("ax_x")
    p["lab_x"] = st.text_input(t("lab_x"), key="lab_x", help=t("lab_help"))
    r1, r2 = st.columns(2)
    xmin = r1.text_input(t("xmin"), key="xmin")
    xmax = r2.text_input(t("xmax"), key="xmax")
    stx = st.text_input(t("stepx"), key="step_x")
    axc_x = st.selectbox(t("ax_color"), AXC, index=0, format_func=lambda k: t("yc_" + k), key="axc_x")
    xb_cust = st.color_picker(t("color_xb"), "#000000", key="xb_col") if axc_x == "custom" else "#000000"

    # ---- 上方 x 轴
    p["lab_x2"], x2min, x2max, stx2 = "", "", "", ""
    axc_x2, xt_cust = "follow", "#000000"
    if p["dual_x"]:
        st.divider()
        _h("ax_x2")
        p["lab_x2"] = st.text_input(t("lab_x2"), key="lab_x2", help=t("lab_help"))
        r1, r2 = st.columns(2)
        x2min = r1.text_input(t("x2min"), key="x2min")
        x2max = r2.text_input(t("x2max"), key="x2max")
        stx2 = st.text_input(t("stepx2"), key="step_x2")
        axc_x2 = st.selectbox(t("ax_color"), AXC, index=1, format_func=lambda k: t("yc_" + k), key="axc_x2")
        xt_cust = st.color_picker(t("color_xt"), "#000000", key="xt_col") if axc_x2 == "custom" else "#000000"

    # ---- 左 y 轴
    st.divider()
    _h("ax_y")
    p["lab_y"] = st.text_input(t("lab_y"), key="lab_y", help=t("lab_help"))
    r1, r2 = st.columns(2)
    ymin = r1.text_input(t("ymin"), key="ymin")
    ymax = r2.text_input(t("ymax"), key="ymax")
    sty = st.text_input(t("stepy"), key="step_y")
    axc_y = st.selectbox(t("ax_color"), AXC, index=(1 if p["dual"] else 0), format_func=lambda k: t("yc_" + k), key=f"axc_y_{p['dual']}")
    yl_cust = st.color_picker(t("color_yl"), "#000000", key="yl_col") if axc_y == "custom" else "#000000"

    # ---- 右 y 轴
    p["lab_y2"], y2min, y2max, sty2 = "", "", "", ""
    axc_y2, yr_cust = "follow", "#000000"
    if p["dual"]:
        st.divider()
        _h("ax_y2")
        p["lab_y2"] = st.text_input(t("lab_y2"), key="lab_y2", help=t("lab_help"))
        r1, r2 = st.columns(2)
        y2min = r1.text_input(t("y2min"), key="y2min")
        y2max = r2.text_input(t("y2max"), key="y2max")
        sty2 = st.text_input(t("stepy2"), key="step_y2")
        axc_y2 = st.selectbox(t("ax_color"), AXC, index=1, format_func=lambda k: t("yc_" + k), key="axc_y2")
        yr_cust = st.color_picker(t("color_yr"), "#000000", key="yr_col") if axc_y2 == "custom" else "#000000"

    st.caption(t("lab_note"))

    # ---- 四个轴共用的刻度设置
    st.divider()
    _h("sub_ticks_common")
    st.caption(t("tick_marks"))
    cc = st.columns(4)
    p["tick_marks"] = {s: cc[i].checkbox(t("side_" + s), value=True, key=f"tm_{s}") for i, s in enumerate(SIDES)}
    st.caption(t("tick_nums"))
    cc = st.columns(4)
    num_default = {"bottom": True, "left": True, "top": p["dual_x"], "right": p["dual"]}
    p["tick_nums"] = {
        s: cc[i].checkbox(t("side_" + s), value=num_default[s], key=f"tn_{s}_{p['dual']}_{p['dual_x']}")
        for i, s in enumerate(SIDES)
    }
    p["minor"] = st.checkbox(t("minor"), value=True, key="minor")
    p["tick_dir"] = st.radio(t("tdir"), ["in", "out"], format_func=lambda d: t("dir_" + d), horizontal=True, key="tdir")

# ================================================================ 4. 曲线与图例
with sb.expander(t("grp_curves"), expanded=True):
    _h("sub_curves")
    p["colors"], p["colors_x2"] = [], []
    for k, name in enumerate(dfs):
        stem = Path(name).stem
        if p["dual"]:
            cl = st.color_picker(t("color_left", name=stem), COLORS[(2 * k) % 7], key=f"cl_d_{name}")
            cr = st.color_picker(t("color_right", name=stem), COLORS[(2 * k + 1) % 7], key=f"cr_d_{name}")
        else:
            cl = st.color_picker(t("color_curve", name=stem), COLORS[k % 7], key=f"cl_s_{name}")
            cr = None
        p["colors"].append((cl, cr))
        if p["dual_x"]:
            cx = st.color_picker(
                t("color_x2curve", name=stem), COLORS[(2 * k + (2 if p["dual"] else 1)) % 7], key=f"cx_{name}_{p['dual']}"
            )
        else:
            cx = None
        p["colors_x2"].append(cx)
    p["fit_color"] = st.color_picker(t("color_fit"), "#000000", key="cfit") if kind != "none" else "#000000"
    p["scatter"] = st.checkbox(t("scatter"), value=(mode != "CV"), key=f"scatter_{mode}")
    p["show_peaks"] = st.checkbox(t("show_peaks"), key="show_peaks") if mode == "CV" else False

    st.divider()
    _h("sub_legend")
    p["legend"] = st.checkbox(t("legend"), value=True, key="legend")
    p["legend_loc"] = st.selectbox(
        t("legend_loc"), ["best", "upper right", "upper left", "lower right", "lower left", "manual"],
        format_func=lambda k: t("leg_manual") if k == "manual" else k, key="leg_loc",
    )
    p["legend_xy"] = (0.02, 0.98)
    if p["legend_loc"] == "manual":
        lc = st.columns(2)
        p["legend_xy"] = (
            lc[0].number_input(t("leg_x"), value=0.02, step=0.05, format="%.2f", key="leg_pos_x"),
            lc[1].number_input(t("leg_y"), value=0.98, step=0.05, format="%.2f", key="leg_pos_y"),
        )
        st.caption(t("leg_manual_note"))
    p["legend_ncol"] = int(st.number_input(t("leg_ncol"), min_value=1, max_value=6, value=1, step=1, key="leg_ncol"))
    p["legend_frame"] = st.checkbox(t("leg_frame"), value=False, key="leg_frame")
    st.caption(t("sec_leg_text"))
    p["custom_legends"], p["custom_legends_x2"] = [], []
    for k, name in enumerate(dfs):
        stem = Path(name).stem
        if p["dual"]:
            l_val = st.text_input(t("leg_text_left", name=stem), key=f"leg_l_{name}_{p['dual']}")
            r_val = st.text_input(t("leg_text_right", name=stem), key=f"leg_r_{name}_{p['dual']}")
            p["custom_legends"].append((l_val.strip(), r_val.strip()))
        else:
            val = st.text_input(t("leg_text_single", name=stem), key=f"leg_s_{name}_{p['dual']}")
            p["custom_legends"].append((val.strip(), ""))
        if p["dual_x"]:
            x_val = st.text_input(t("leg_text_x2", name=stem), key=f"leg_x_{name}")
            p["custom_legends_x2"].append(x_val.strip())
        else:
            p["custom_legends_x2"].append("")

# 轴的颜色：三选一（黑色 / 跟随曲线 / 自定义）。要等曲线颜色确定后才能解析“跟随”
def _axis_color(mode_, custom, follow):
    return "#000000" if mode_ == "black" else (follow if mode_ == "follow" else custom)


_c0l, _c0r = p["colors"][0]
p["xb_col"] = _axis_color(axc_x, xb_cust, _c0l)
p["yl_col"] = _axis_color(axc_y, yl_cust, _c0l)
p["yr_col"] = _axis_color(axc_y2, yr_cust, _c0r or "#000000") if p["dual"] else "#000000"
p["xt_col"] = _axis_color(axc_x2, xt_cust, p["colors_x2"][0] or "#000000") if p["dual_x"] else "#000000"

# ================================================================ 5. 图形外观
with sb.expander(t("grp_fig")):
    p["w"] = st.selectbox(t("width"), [85, 120, 175], format_func=lambda w: t(f"w{w}"), key="width")
    p["h"] = st.slider(t("height"), 40, 120, 65, key="height")
    p["fs"] = st.slider(t("fontsize"), 6, 14, 9, key="fs")

# ================================================================ 6. AI 分析设置
with sb.expander(t("sec_ai")):
    ai_key_typed = st.text_input(t("ai_key"), type="password", key="ai_key", help=t("ai_key_help"))
    ai_model = st.text_input(t("ai_model"), value="gemini-3.8-flash", key="ai_model", help=t("ai_model_help"))
    ai_ctx = st.text_area(t("ai_ctx"), key="ai_ctx", placeholder=t("ai_ctx_ph"), height=80)
    ai_scan = st.text_input(t("ai_scan"), key="ai_scan") if mode == "CV" else ""
    ai_q = st.text_area(t("ai_q"), key="ai_q", height=70)
    ai_curve = st.checkbox(t("ai_sendcurve"), key="ai_curve")

# ---------------------------------------------------------------- 处理逻辑
try:
    bad_idx = parse_idx(ex_idx)
    bad_ranges = parse_ranges(ex_range)
    fit_ranges = parse_ranges(fit_range)
except ValueError:
    st.warning(t("warn_range"))
    st.stop()
try:
    p["xlim"] = (parse_float(xmin), parse_float(xmax))
    p["ylim"] = (parse_float(ymin), parse_float(ymax))
    p["y2lim"] = (parse_float(y2min), parse_float(y2max))
    p["x2lim"] = (parse_float(x2min), parse_float(x2max))
    p["xstep"], p["ystep"], p["x2step"], p["y2step"] = parse_float(stx), parse_float(sty), parse_float(stx2), parse_float(sty2)
    if any(v is not None and v <= 0 for v in (p["xstep"], p["ystep"], p["x2step"], p["y2step"])):
        raise ValueError("step")
except ValueError:
    st.warning(t("warn_num"))
    st.stop()

results, rows = [], []
for name, df in dfs.items():
    d = df.iloc[int(p_start) : (int(p_end) or None)]
    idx = d.index.to_numpy().copy()
    x, y_raw = transform(d.iloc[:, xi].to_numpy(float).copy(), d.iloc[:, yi].to_numpy(float).copy(), mode, p)
    
    if mode != "CV":
        x = time_conv(x, p["time_unit"])
        
    y2_raw = transform_y2(d.iloc[:, y2i].to_numpy(float).copy(), y2kind, p) if p["dual"] else None
    x2_raw = transform_x2(d.iloc[:, x2i].to_numpy(float).copy(), x2kind, p) if p["dual_x"] else None
    keep = np.isfinite(x) & np.isfinite(y_raw)
    if y2_raw is not None:
        keep &= np.isfinite(y2_raw)
    if x2_raw is not None:
        keep &= np.isfinite(x2_raw)
    if use_hampel:
        keep &= ~hampel(y_raw, win, nsig)
    for lo, hi in bad_ranges:
        keep &= ~((x >= lo) & (x <= hi))
    keep &= ~np.isin(idx, list(bad_idx))
    
    x2 = x2_raw
    if p["reset_mode"] != "off" and keep.any():
        idx = idx.copy()
        if p["reset_mode"] == "all":
            idx[keep] = np.arange(keep.sum())
            fix = renumber_x
        else:
            idx[keep] = idx[keep] - idx[keep][0]
            fix = shift_head
        # 递补只针对时间轴：CV 的横坐标是电位，不能平移
        if mode != "CV":
            x = fix(x, keep)
        if x2 is not None and x2kind == "time":
            x2 = fix(x2, keep)

    y, y2 = y_raw.copy(), (y2_raw.copy() if y2_raw is not None else None)
    if p["sm_method"] != "none":
        y[keep] = smooth_1d(y_raw[keep], p["sm_method"], sm_win, sm_poly, sm_sigma)
        if y2 is not None and p["sm_right"]:
            y2[keep] = smooth_1d(y2_raw[keep], p["sm_method"], sm_win, sm_poly, sm_sigma)

    fit, fit_info = None, None
    row = {t("tbl_file"): Path(name).stem, t("tbl_removed"): int((~keep).sum())}
    if kind != "none":
        m = keep.copy()
        if fit_ranges:
            m &= (x >= fit_ranges[0][0]) & (x <= fit_ranges[0][1])
        if kind == "cottrell":
            m &= x > 0
        try:
            func, names, vals, err, r2 = do_fit(x[m], y_raw[m], kind, deg)
            xx = np.linspace(x[m].min(), x[m].max(), 400)
            fit = (xx, func(xx), r2)
            fit_info = {
                "model": kind,
                "parameters": {n: _sig(v) for n, v in zip(names, vals)},
                "std_errors": {n: _sig(e, 2) for n, e in zip(names, err)},
                "R2": _sig(r2, 5),
                "x_range_used": [_sig(x[m].min()), _sig(x[m].max())],
            }
            row[t("tbl_params")] = "; ".join(f"{n} = {v:.4g} ± {e:.2g}" for n, v, e in zip(names, vals, err))
            row[t("tbl_r2")] = round(float(r2), 5)
        except Exception as e:
            row[t("tbl_params")] = t("fit_fail", err=e)
    rows.append(row)
    results.append(
        {
            "name": Path(name).stem,
            "idx": idx,
            "x": x,
            "y": y,
            "y_raw": y_raw,
            "y2": y2,
            "y2_raw": y2_raw,
            "keep": keep,
            "fit": fit,
            "fit_info": fit_info,
            "ylab": short(cols[yi]),
            "y2lab": short(cols[y2i]),
            "y2name": cols[y2i],
            "x2": x2,
            "x2name": cols[x2i],
        }
    )

analysis = []
for r in results:
    a = {
        "name": r["name"],
        "n_points_used": int(r["keep"].sum()),
        "n_points_removed": int((~r["keep"]).sum()),
        "x_range": [_sig(np.nanmin(r["x"][r["keep"]])), _sig(np.nanmax(r["x"][r["keep"]]))] if r["keep"].any() else None,
        "y_range": [_sig(np.nanmin(r["y"][r["keep"]])), _sig(np.nanmax(r["y"][r["keep"]]))] if r["keep"].any() else None,
    }
    if mode == "CV":
        a["cv_peaks"] = cv_summary(r["x"], r["y"], r["keep"])
    else:
        a["trace"] = trace_summary(r["x"], r["y"], r["keep"])
    if r["fit_info"]:
        a["fit"] = r["fit_info"]
    if ai_curve:
        a["curve_downsampled_xy"] = downsample(r["x"], r["y"], r["keep"])
    analysis.append(a)
for r, a in zip(results, analysis):
    r["peaks"] = a.get("cv_peaks")

# ---------------------------------------------------------------- 输出与视图渲染
left, right = st.columns([3, 2])
with left:
    tab_s, tab_i, tab_t = st.tabs([t("tab_static"), t("tab_inter"), t("tab_table")])
    with tab_s:
        prev = build_fig(results, mode, p, show_removed=True)
        if getattr(prev, "_tick_skipped", False):
            st.warning(t("warn_step"))
        st.image(export_bytes(prev, "png", 250), width=640, caption=t("preview_caption"))
        plt.close(prev)
    with tab_i:
        if not HAS_PLOTLY:
            st.warning(t("no_plotly"))
        else:
            h1, h2, h3 = st.columns(3)
            hv = h1.radio(t("hover_mode"), ["closest", "x"], format_func=lambda k: t("hv_" + k), key="hover_mode")
            spikes = h2.checkbox(t("crosshair"), value=True, key="spikes")
            show_rm = h3.checkbox(t("show_removed"), value=True, key="show_removed_i")
            st.plotly_chart(
                build_plotly(results, mode, p, hv, spikes, show_rm), theme=None,
                config={"scrollZoom": True, "displaylogo": False}, key="plotly_main",
            )
            st.caption(t("inter_hint"))
    with tab_t:
        st.caption(t("tbl_hint"))
        only_kept = st.checkbox(t("tbl_only_kept"), value=True, key="tbl_only_kept")
        names = [r["name"] for r in results]
        pick = st.selectbox(t("tbl_file_sel"), range(len(names)), format_func=lambda i: names[i], key="tbl_pick") if len(names) > 1 else 0
        pdf = processed_frame(results[pick], mode, p, only_kept)
        st.dataframe(pdf, hide_index=True, height=360)
        st.download_button(
            t("tbl_dl"), pdf.to_csv(index=False).encode("utf-8-sig"), file_name=f"{names[pick]}_processed.csv",
            mime="text/csv", key="tbl_dl",
        )
with right:
    st.subheader(t("results"))
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    if mode == "CV":
        st.subheader(t("sec_peaks"))
        prow = []
        for a in analysis:
            c = a.get("cv_peaks") or {}
            prow.append(
                {
                    t("tbl_file"): a["name"],
                    "Epa": c.get("Epa"), "ipa": c.get("ipa"), "Epc": c.get("Epc"), "ipc": c.get("ipc"),
                    "ΔEp / mV": c.get("dEp_mV"), "E1/2": c.get("E_half"), "ipa/|ipc|": c.get("ipa_over_abs_ipc"),
                }
            )
        if any(len(a.get("cv_peaks") or {}) > 1 for a in analysis):
            st.dataframe(pd.DataFrame(prow), hide_index=True)
        else:
            st.caption(t("pk_none"))
        st.caption(t("pk_note"))

st.divider()
st.subheader(t("sec_export"))
e1, e2, e3, e4 = st.columns([1, 1, 2, 2])
fmt = e1.selectbox(t("fmt"), ["png", "pdf", "svg", "tiff"], format_func=str.upper, key="fmt")
dpi = e2.selectbox(t("dpi"), [300, 600, 1200], index=1, key="dpi")
fname = e3.text_input(t("fname"), value="figure", key="fname")
final = build_fig(results, mode, p, show_removed=False)
payload = export_bytes(final, fmt, dpi)
plt.close(final)
e4.markdown("&nbsp;")
e4.download_button(t("dl"), payload, file_name=f"{fname.strip() or 'figure'}.{fmt}", mime=MIME[fmt], type="primary", key="dl")
st.caption(t("dl_hint"))

# ---------------------------------------------------------------- AI 分析
st.divider()
st.subheader(t("sec_ai"))
st.caption(t("ai_intro"))
st.caption(t("ai_warn"))

xl_auto, yl_auto = axis_labels(mode, p)
payload = {
    "technique": {"CV": "cyclic voltammetry", "CA": "chronoamperometry (constant potential)", "CP": "chronopotentiometry (constant current)"}[mode],
    "axes": {"x": p["lab_x"] or xl_auto, "y_left": p["lab_y"] or yl_auto},
    "processing": {
        "reference_electrode": None if p["ref"] == "none" else p["ref"],
        "converted_to_RHE": bool(p["to_rhe"]),
        "current_normalized_by_area_cm2": p["area"] if p["norm"] else None,
        "outlier_removal": {"hampel_window": win, "hampel_sigma": nsig} if use_hampel else None,
        "excluded_x_ranges": ex_range or None,
        "excluded_rows": ex_idx or None,
        "smoothing": None if p["sm_method"] == "none" else {"method": p["sm_method"], "window": sm_win, "polyorder": sm_poly, "sigma": sm_sigma},
        "fit_uses": "cleaned, unsmoothed data",
    },
    "files": analysis,
    "user_context": ai_ctx.strip() or None,
    "scan_rate_mV_per_s": ai_scan.strip() or None,
}
if p["dual"]:
    payload["axes"]["y_right"] = p["lab_y2"] or y2_label(p, p["y2kind"], results[0]["y2name"])
if p["dual_x"]:
    payload["axes"]["x_top"] = p["lab_x2"] or x2_label(p, p["x2kind"], results[0]["x2name"])
payload_json = json.dumps(payload, ensure_ascii=False, indent=1, default=str)
question = ai_q.strip() or "Please analyze this data."
user_text = f"{question}\n\nData summary (JSON):\n\n```json\n{payload_json}\n```"
system_text = AI_SYSTEM.format(language=AI_LANG[lang])
sig_now = hashlib.sha1((ai_model + system_text + user_text).encode("utf-8")).hexdigest()

with st.expander(t("ai_preview")):
    st.code(payload_json, language="json")

if st.button(t("ai_btn"), type="primary", key="ai_go"):
    api_key = get_api_key(ai_key_typed)
    if not api_key:
        st.error(t("ai_nokey"))
    else:
        try:
            import google.genai
        except ImportError:
            st.error(t("ai_nopkg"))
        else:
            try:
                with st.spinner(t("ai_spin")):
                    text = call_gemini(
                        api_key,
                        ai_model.strip(),
                        system_text,
                        user_text
                    )
                st.session_state["ai_result"] = (sig_now, text)
            except Exception as e:
                st.error(t("ai_fail", err=f"{type(e).__name__}: {str(e)[:300]}"))

saved = st.session_state.get("ai_result")
if saved:
    if saved[0] != sig_now:
        st.caption(t("ai_stale"))
    st.markdown(saved[1])