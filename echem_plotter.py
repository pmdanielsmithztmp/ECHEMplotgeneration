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
CUR = {"A": 1.0, "mA": 1e-3, "µA": 1e-6, "nA": 1e-9, "pA": 1e-12}
REF = {
    "SHE": 0.000,
    "Ag/AgCl (sat. KCl)": 0.197,
    "SCE": 0.241,
    "Hg/HgO (1 M NaOH)": 0.098,
    "Hg/Hg₂SO₄": 0.615,
}
FITS = {
    "CV": ["none", "linear", "poly"],
    "CA": ["none", "cottrell", "exp", "linear", "poly"],
    "CP": ["none", "linear", "poly", "exp"],
}
MIME = {"png": "image/png", "pdf": "application/pdf", "svg": "image/svg+xml", "tiff": "image/tiff"}
SIDES = ["bottom", "left", "top", "right"]

# 字体设置
_LATIN = ["Times New Roman", "Liberation Serif", "Nimbus Roman", "TeX Gyre Termes", "STIXGeneral"]
_CJK_JA = ["Yu Mincho", "MS Mincho", "MS PMincho", "Hiragino Mincho ProN", "IPAexMincho", "IPAMincho", "Noto Serif CJK JP"]
_CJK_ZH = ["SimSun", "NSimSun", "Songti SC", "STSong", "Noto Serif CJK SC", "Source Han Serif SC"]
_AVAIL = {f.name for f in font_manager.fontManager.ttflist}
FONTS_BY_LANG = {
    "ja": [f for f in _LATIN + _CJK_JA + _CJK_ZH if f in _AVAIL] + ["DejaVu Serif"],
    "zh": [f for f in _LATIN + _CJK_ZH + _CJK_JA if f in _AVAIL] + ["DejaVu Serif"],
}
FONTS = FONTS_BY_LANG["ja"]

# ---------------------------------------------------------------- 多语言
TR = {
    "zh": {
        "title": "电化学论文曲线图生成器",
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
        "grp_textbox": "自定义文本框 / 标注",
        "grp_fig": "图形外观",
        "sub_cols": "数据列",
        "sub_rows": "使用的行范围",
        "sub_curves": "曲线颜色",
        "sub_legend": "图例",
        "num_tb": "文本框数量",
        "tb_title": "文本框 #{n}",
        "tb_text": "文本内容（支持 $E_{{pa}}$ 等公式）",
        "tb_coord": "坐标类型",
        "coord_data": "数据坐标 (Data)",
        "coord_rel": "相对坐标 (0 左/下 -> 1 右/上)",
        "tb_x": "x 坐标",
        "tb_y": "y 坐标",
        "tb_fs": "字号 / pt",
        "tb_color": "文本颜色",
        "tb_frame": "边框样式",
        "frame_none": "无边框透明背景",
        "frame_box": "圆角框 (半透明背景)",
        "frame_solid": "方框 (白色实背景)",
        "tb_rot": "旋转角度 / °",
        "ax_x": "下方 x 轴",
        "ax_x2": "上方 x 轴 (x2)",
        "ax_y": "左 y 轴",
        "ax_y2": "右 y 轴 (y2)",
        "ax_color": "轴的颜色（标签、刻度、轴线）",
        "color_xb": "下方 x 轴颜色",
        "sub_ticks_common": "刻度设置（四个轴共用）",
        "color_x2curve": "{name} 上方 x 轴曲线颜色",
        "leg_text_x2": "{name} (上方 x 轴) 图例名称",
        "detect_note": "自动识别：x = {x}；y = {y}",
        "role_mismatch": "{which} 看起来是“{got}”，而这个曲线类型需要“{want}”，请确认选对了列。",
        "role_time": "时间",
        "role_pot": "电位/电压",
        "role_cur": "电流",
        "role_charge": "电量/容量",
        "role_idx": "序号/工步",
        "role_other": "其他",
        "unit_auto": "自动（按列名）",
        "time_raw_unit": "原始时间单位",
        "unit_pot_raw": "原始电位单位",
        "unit_auto_note": "按列名识别到：{info}。列名没写单位的按 s / A / V 处理，也可以手动指定。",
        "q_seg": "电量分段方式",
        "q_seg_auto": "自动分段（电流正负号 / CV 扫描方向变化处，每段从 0 累计）",
        "q_seg_none": "不分段（整个文件连续累计）",
        "cp_src": "CP 的电流来源",
        "cp_src_col": "数据里的电流列",
        "cp_src_const": "恒定电流（手动输入）",
        "cp_col": "电流列",
        "cp_cur_raw": "电流列的原始单位",
        "cp_const": "恒定电流 / mA",
        "cv_time_note": "CV：文件里有时间列就用它积分，没有就按扫速和电位变化推算时间。",
        "calc_note": "只用保留下来的点积分，被剔除的点和 NaN 不参与；时间不增加的地方会断开，不会算成负电量；每个分段起点处的一个采样间隔不计入。",
        "calc_err_few": "保留的点少于 2 个",
        "calc_err_no_time": "找不到时间信息（CA / CP 需要时间列）",
        "calc_fail": "{name}：电量计算失败：{err}",
        "q_summary": "电量 / 比容量汇总（每个分段）",
        "qs_file": "文件",
        "qs_seg": "分段",
        "qs_dir": "方向",
        "qs_n": "点数",
        "qs_dur": "时长 / s",
        "qs_q": "Q / C",
        "qs_cap": "容量",
        "qs_none": "没有可汇总的分段。",
        "skip_rows": "{name}：有 {n} 行列数对不上，已跳过。",
        "time_unit": "显示的时间单位（原始单位见上一项）",
        "pot_target": "电位基准 / 换算目标",
        "pot_raw": "保持原始电位 (vs. 参比)",
        "to_she": "换算为 SHE (vs. SHE)",
        "to_rhe": "换算为 RHE (vs. RHE)",
        "custom_offset": "自定义电位偏移量 / V",
        "ref_custom": "自定义参比电极…",
        "ref_custom_name": "自定义参比电极名称",
        "ref_custom_E": "该参比电极相对 SHE 的电位 / V",
        "reset_mode": "剔除点后坐标递补",
        "reset_all": "全部收拢",
        "reset_head": "只收拢开头",
        "reset_off": "不递补",
        "reset_note": "递补只作用于时间轴。",
        "sec_q_capacity": "电量与比容量计算选项",
        "calc_q": "生成电量图 (Accumulated Charge Q)",
        "calc_cap": "生成比容量图 (Specific Capacity)",
        "abs_q": "按绝对值积分（每段电量 / 容量取正值）",
        "scan_rate": "扫速 v / (mV s⁻¹)",
        "active_mass": "活性物质质量 / g（0 表示不进行质量归一化）",
        "leg_manual": "手动指定位置",
        "leg_x": "图例 x（0 左 → 1 右）",
        "leg_y": "图例 y（0 下 → 1 上）",
        "leg_manual_note": "以图例左上角为基准点。",
        "leg_ncol": "图例列数",
        "leg_frame": "显示图例边框",
        "x2_note": "上方 x 轴设置。",
        "xcol": "x 列",
        "ycol": "y 列（左轴）",
        "start": "起始行",
        "end": "结束行（0=到末尾）",
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
        "ph": "溶液 pH",
        "sec_clean": "坏点剔除",
        "hampel": "自动检测离群点（Hampel）",
        "win": "窗口（点）",
        "hampel_note": "调整窗口与阈值剔除离群尖峰。",
        "nsig": "阈值（σ）",
        "ex_range": "按 x 范围剔除",
        "ex_idx": "按行号剔除",
        "sec_smooth": "平滑",
        "sm_method": "平滑方法",
        "sm_none": "不平滑",
        "sm_savgol": "Savitzky–Golay",
        "sm_ma": "移动平均",
        "sm_gauss": "高斯滤波",
        "sm_win": "窗口（点）",
        "sm_poly": "多项式阶数",
        "sm_sigma": "σ（点）",
        "sm_right": "同时平滑右轴曲线",
        "sm_raw": "同时显示原始数据（浅色）",
        "sm_note": "拟合使用的是未平滑的原始数据。",
        "sec_fit": "拟合（针对左轴曲线）",
        "fit_model": "模型",
        "deg": "多项式阶数",
        "fit_range": "拟合范围",
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
        "warn_num": "坐标轴范围和刻度间隔需要填写数字。",
        "step_title": "主刻度间隔",
        "stepx": "x 间隔",
        "stepy": "y 间隔",
        "stepy2": "右 y 间隔",
        "stepx2": "上方 x 间隔",
        "warn_step": "主刻度间隔太小，已忽略。",
        "lab_note": "支持 LaTeX 公式语法，如 `$E_{pa}$`。",
        "sec_ticks": "刻度显示",
        "tick_marks": "刻度线",
        "tick_nums": "刻度数字",
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
        "yaxis_color": "y 轴颜色",
        "yc_black": "黑色",
        "yc_follow": "跟随曲线颜色",
        "yc_custom": "自定义",
        "color_yl": "左 y 轴颜色",
        "color_yr": "右 y 轴颜色",
        "sec_leg_text": "自定义图例名称",
        "leg_text_single": "{name} 图例名称",
        "leg_text_left": "{name} (左轴) 图例名称",
        "leg_text_right": "{name} (右轴) 图例名称",
        "sec_labels": "坐标轴标签",
        "lab_x": "x 轴标签",
        "lab_x2": "上方 x 轴标签",
        "lab_y": "左 y 轴标签",
        "lab_y2": "右 y 轴标签",
        "lab_help": "支持 LaTeX 公式。",
        "sec_ai": "AI 分析",
        "ai_intro": "把处理后的数据摘要发给 Gemini 辅助分析。",
        "ai_warn": "仅供参考。",
        "ai_key": "API 密钥",
        "ai_key_help": "仅在内存中保存。",
        "ai_model": "模型",
        "ai_model_help": "Google Gemini 模型名称。",
        "ai_ctx": "实验背景（可选）",
        "ai_ctx_ph": "例如：0.1 M PBS，检测葡萄糖",
        "ai_scan": "扫速 / mV s⁻¹",
        "ai_q": "问题（可选）",
        "ai_sendcurve": "发送降采样数据",
        "ai_preview": "查看发送给 AI 的内容",
        "ai_btn": "生成 AI 分析",
        "ai_spin": "AI 分析中……",
        "ai_nokey": "请填写 API 密钥。",
        "ai_nopkg": "未安装 google-genai 包。",
        "ai_fail": "AI 调用失败：{err}",
        "ai_stale": "（参数已修改，请重新生成。）",
        "sec_peaks": "CV 峰分析",
        "pk_note": "未做基线校正。",
        "pk_none": "未检测到明显的峰。",
        "tab_static": "静态预览",
        "tab_inter": "交互查看",
        "tab_table": "处理后的数据",
        "hover_mode": "悬停方式",
        "hv_closest": "最近点",
        "hv_x": "同一 x 轴",
        "crosshair": "十字准线",
        "show_removed": "显示剔除点（红色 ×）",
        "inter_hint": "悬停查看数据。",
        "no_plotly": "未安装 plotly。",
        "show_peaks": "标注峰位",
        "tbl_only_kept": "只保留未剔除点",
        "tbl_file_sel": "选择文件",
        "tbl_dl": "下载处理数据 (CSV)",
        "tbl_hint": "处理后数据可直接导入 Origin。",
        "warn_range": "范围格式错误。",
        "preview_caption": "预览图",
        "results": "结果",
        "tbl_file": "文件",
        "tbl_removed": "剔除点数",
        "tbl_params": "参数",
        "tbl_r2": "R²",
        "fit_fail": "拟合失败：{err}",
        "sec_export": "输出与保存图片",
        "fmt": "格式",
        "dpi": "分辨率 (dpi)",
        "fname": "文件名前缀",
        "dl": "下载主图",
        "dl_q": "下载电量图",
        "dl_cap": "下载比容量图",
        "dl_hint": "矢量图 (PDF/SVG) / 位图 (PNG/TIFF)。",
    },
    "ja": {
        "title": "電気化学 論文用グラフ作成ツール",
        "mode_label": "測定の種類",
        "mode_CV": "CV サイクリックボルタンメトリー",
        "mode_CA": "CA クロノアンペロメトリー",
        "mode_CP": "CP クロノポテンショメトリー",
        "upload": "データをアップロード",
        "use_demo": "サンプル数据を使用",
        "enc": "文字コード",
        "enc_auto": "自動判定",
        "enc_utf8": "UTF-8",
        "enc_gbk": "GBK",
        "enc_sjis": "Shift-JIS",
        "info_upload": "ファイルをアップロードしてください。",
        "demo_note": "サンプルデータを表示中。",
        "parse_fail": "{name} 読み込み失敗：{err}",
        "err_two_cols": "2列以上必要です。",
        "sec_data": "数据列",
        "grp_data": "数据",
        "grp_proc": "数据处理",
        "grp_axes": "轴",
        "grp_curves": "曲线",
        "grp_textbox": "テキスト",
        "grp_fig": "外観",
        "sub_cols": "列",
        "sub_rows": "行範囲",
        "sub_curves": "色",
        "sub_legend": "凡例",
        "num_tb": "テキスト数",
        "tb_title": "テキスト #{n}",
        "tb_text": "内容",
        "tb_coord": "座標",
        "coord_data": "数据座標",
        "coord_rel": "相対座標",
        "tb_x": "x 座標",
        "tb_y": "y 座標",
        "tb_fs": "サイズ",
        "tb_color": "文字色",
        "tb_frame": "枠",
        "frame_none": "なし",
        "frame_box": "半透明",
        "frame_solid": "白背景",
        "tb_rot": "回転",
        "ax_x": "下 x 轴",
        "ax_x2": "上 x 轴",
        "ax_y": "左 y 轴",
        "ax_y2": "右 y 轴",
        "ax_color": "轴の色",
        "color_xb": "下 x 轴色",
        "sub_ticks_common": "目盛り",
        "color_x2curve": "{name} 上 x 轴色",
        "leg_text_x2": "{name} 凡例",
        "detect_note": "自動認識：x = {x}；y = {y}",
        "role_mismatch": "{which} は「{got}」のようですが、この測定には「{want}」が必要です。列の選択を確認してください。",
        "role_time": "時間",
        "role_pot": "電位/電圧",
        "role_cur": "電流",
        "role_charge": "電荷量/容量",
        "role_idx": "番号/ステップ",
        "role_other": "その他",
        "unit_auto": "自動（列名から）",
        "time_raw_unit": "元データの時間単位",
        "unit_pot_raw": "元データの電位単位",
        "unit_auto_note": "列名から認識：{info}。単位の書かれていない列は s / A / V として扱います。手動指定もできます。",
        "q_seg": "電荷量の区間分け",
        "q_seg_auto": "自動（電流の正負 / CV の掃引方向が変わる所で区切り、各区間を 0 から積算）",
        "q_seg_none": "区切らない（ファイル全体を連続で積算）",
        "cp_src": "CP の電流の取得元",
        "cp_src_col": "データ中の電流列",
        "cp_src_const": "定電流（手入力）",
        "cp_col": "電流列",
        "cp_cur_raw": "電流列の元の単位",
        "cp_const": "定電流 / mA",
        "cv_time_note": "CV：ファイルに時間列があればそれで積分し、なければ掃引速度と電位変化から時間を推定します。",
        "calc_note": "残した点だけで積分します（除去した点と NaN は使いません）。時間が増えない所では区切り、負の電荷量にはなりません。各区間の先頭の 1 サンプル間隔は含まれません。",
        "calc_err_few": "残った点が 2 点未満です",
        "calc_err_no_time": "時間情報が見つかりません（CA / CP には時間列が必要です）",
        "calc_fail": "{name}：電荷量の計算に失敗：{err}",
        "q_summary": "電荷量 / 比容量のまとめ（区間ごと）",
        "qs_file": "ファイル",
        "qs_seg": "区間",
        "qs_dir": "方向",
        "qs_n": "点数",
        "qs_dur": "時間 / s",
        "qs_q": "Q / C",
        "qs_cap": "容量",
        "qs_none": "まとめる区間がありません。",
        "skip_rows": "{name}：列数が合わない {n} 行をスキップしました。",
        "time_unit": "表示する時間の単位",
        "pot_target": "電位基準",
        "pot_raw": "換算なし",
        "to_she": "SHE 換算",
        "to_rhe": "RHE 換算",
        "custom_offset": "電位オフセット / V",
        "ref_custom": "カスタム参照電極…",
        "ref_custom_name": "参照電極名",
        "ref_custom_E": "SHE 基準電位 / V",
        "reset_mode": "詰める",
        "reset_all": "すべて",
        "reset_head": "先頭のみ",
        "reset_off": "なし",
        "reset_note": "時間軸のみ。",
        "sec_q_capacity": "電荷量・比容量オプション",
        "calc_q": "電荷量グラフ生成",
        "calc_cap": "比容量グラフ生成",
        "abs_q": "絶対値で積算（各区間の値を正にする）",
        "scan_rate": "掃引速度 / (mV s⁻¹)",
        "active_mass": "活物質質量 / g",
        "leg_manual": "手動位置",
        "leg_x": "凡例 x",
        "leg_y": "凡例 y",
        "leg_manual_note": "相対位置。",
        "leg_ncol": "列数",
        "leg_frame": "枠線",
        "x2_note": "上 x 轴。",
        "xcol": "x 列",
        "ycol": "y 列",
        "start": "開始行",
        "end": "終了行",
        "sec_dual": "二轴",
        "dual": "右 y 轴",
        "dual_x": "上 x 轴",
        "y2col": "右轴列",
        "y2kind": "右轴種類",
        "x2col": "上轴列",
        "x2kind": "上轴種類",
        "color_xt": "上轴色",
        "kind_cur": "電流",
        "kind_pot": "電位",
        "kind_time": "時間",
        "kind_other": "その他",
        "sec_units": "単位",
        "cur_raw": "元電流単位",
        "cur_out": "出力単位",
        "norm": "密度にする",
        "area": "面積 / cm²",
        "ref": "参照電極",
        "ref_none": "なし",
        "ph": "pH",
        "sec_clean": "外れ値",
        "hampel": "Hampel",
        "win": "ウィンドウ",
        "hampel_note": "閾値調整。",
        "nsig": "閾値",
        "ex_range": "x 範囲除外",
        "ex_idx": "行番号除外",
        "sec_smooth": "平滑化",
        "sm_method": "方法",
        "sm_none": "なし",
        "sm_savgol": "Savitzky–Golay",
        "sm_ma": "移動平均",
        "sm_gauss": "ガウシアン",
        "sm_win": "幅",
        "sm_poly": "次数",
        "sm_sigma": "σ",
        "sm_right": "右轴も",
        "sm_raw": "元数据表示",
        "sm_note": "元数据を使用。",
        "sec_fit": "フィット",
        "fit_model": "モデル",
        "deg": "次数",
        "fit_range": "範囲",
        "fit_none": "なし",
        "fit_linear": "直線",
        "fit_poly": "多項式",
        "fit_exp": "指数",
        "fit_cottrell": "Cottrell",
        "sec_fig": "外観",
        "width": "幅",
        "w85": "85 mm",
        "w120": "120 mm",
        "w175": "175 mm",
        "height": "高さ / mm",
        "fontsize": "サイズ / pt",
        "scatter": "点表示",
        "legend": "凡例",
        "legend_loc": "位置",
        "sec_range": "轴範囲",
        "xmin": "x 最小",
        "xmax": "x 最大",
        "ymin": "y 最小",
        "ymax": "y 最大",
        "y2min": "右 y 最小",
        "y2max": "右 y 最大",
        "x2min": "上 x 最小",
        "x2max": "上 x 最大",
        "warn_num": "数値を入力。",
        "step_title": "目盛り間隔",
        "stepx": "x 間隔",
        "stepy": "y 間隔",
        "stepy2": "右 y 間隔",
        "stepx2": "上 x 間隔",
        "warn_step": "間隔が小。",
        "lab_note": "LaTeX 数式対応。",
        "sec_ticks": "目盛り",
        "tick_marks": "目盛り線",
        "tick_nums": "数値",
        "side_bottom": "下",
        "side_left": "左",
        "side_top": "上",
        "side_right": "右",
        "minor": "補助目盛り",
        "tdir": "向き",
        "dir_in": "内",
        "dir_out": "外",
        "sec_color": "色",
        "color_curve": "{name} 曲線の色",
        "color_left": "{name} 左轴色",
        "color_right": "{name} 右轴色",
        "color_fit": "フィット色",
        "yaxis_color": "y 轴色",
        "yc_black": "黒",
        "yc_follow": "曲線色",
        "yc_custom": "カスタム",
        "color_yl": "左 y 色",
        "color_yr": "右 y 色",
        "sec_leg_text": "凡例名変更",
        "leg_text_single": "{name} 凡例",
        "leg_text_left": "{name} 左 凡例",
        "leg_text_right": "{name} 右 凡例",
        "sec_labels": "轴ラベル",
        "lab_x": "x 轴ラベル",
        "lab_x2": "上 x 轴ラベル",
        "lab_y": "左 y 轴ラベル",
        "lab_y2": "右 y 轴ラベル",
        "lab_help": "LaTeX 対応。",
        "sec_ai": "AI 分析",
        "ai_intro": "Gemini による解析。",
        "ai_warn": "参考用。",
        "ai_key": "API キー",
        "ai_key_help": "メモリ保存。",
        "ai_model": "モデル",
        "ai_model_help": "Gemini モデル名。",
        "ai_ctx": "背景",
        "ai_ctx_ph": "例：0.1 M PBS",
        "ai_scan": "掃引速度",
        "ai_q": "質問",
        "ai_sendcurve": "データ送信",
        "ai_preview": "確認",
        "ai_btn": "実行",
        "ai_spin": "分析中…",
        "ai_nokey": "キーを入力。",
        "ai_nopkg": "パッケージなし。",
        "ai_fail": "失敗：{err}",
        "ai_stale": "（過去の設定）",
        "sec_peaks": "CV ピーク",
        "pk_note": "補正なし。",
        "pk_none": "ピークなし。",
        "tab_static": "プレビュー",
        "tab_inter": "インタラクティブ",
        "tab_table": "データ",
        "hover_mode": "ホバー",
        "hv_closest": "最近点",
        "hv_x": "同一 x",
        "crosshair": "十字線",
        "show_removed": "除外点表示",
        "inter_hint": "ホバー読み取り。",
        "no_plotly": "plotly なし。",
        "show_peaks": "ピーク表示",
        "tbl_only_kept": "除外点除く",
        "tbl_file_sel": "ファイル選択",
        "tbl_dl": "ダウンロード",
        "tbl_hint": "Origin 用。",
        "warn_range": "書式エラー。",
        "preview_caption": "プレビュー",
        "results": "結果",
        "tbl_file": "ファイル",
        "tbl_removed": "除外数",
        "tbl_params": "パラメータ",
        "tbl_r2": "R²",
        "fit_fail": "失敗：{err}",
        "sec_export": "画像出力・保存",
        "fmt": "形式",
        "dpi": "解像度",
        "fname": "ファイル名",
        "dl": "主图ダウンロード",
        "dl_q": "電荷量图ダウンロード",
        "dl_cap": "比容量图ダウンロード",
        "dl_hint": "ベクター/ビットマップ",
    },
}


# ---------------------------------------------------------------- 界面文字修正：中文只用简体中文，日文全部用日语写法
TR["zh"].update({
    "tb_text": "文本内容（支持 $E_{pa}$ 等公式）",
    "x2_note": "上方 x 轴：以所选列为横坐标，把左 y 轴的数据再画一条曲线（颜色、图例名称可单独设置）。",
    "reset_note": "递补只作用于时间轴，CV 的电位不会被改动。",
    "leg_manual_note": "以图例左上角为基准点，坐标是相对绘图区的比例，可以超出 0–1 把图例放到图外。",
    "active_mass": "活性物质质量（0 表示不进行质量归一化）",
    "mass_unit": "质量单位",
    "q_fig_title": "电量图 (Charge Q)",
    "cap_fig_title": "比容量对比图 (Specific Capacity)",
    "fig_title": "图标题（可选）",
    "follow_main": "曲线颜色跟随主图",
    "fig_main": "主图",
    "fig_q": "电量图",
    "fig_cap": "比容量图",
    "grp_qfig": "电量图设置",
    "grp_capfig": "比容量图设置",
    "grp_reflines": "参考线（虚线）",
    "rl_n": "参考线数量",
    "rl_title": "参考线 #{n}",
    "rl_orient": "方向",
    "rl_orient_h": "水平线 (y = 定值)",
    "rl_orient_v": "竖直线 (x = 定值)",
    "rl_pos": "位置（数据坐标）",
    "rl_color": "线条颜色",
    "rl_style": "线型",
    "rl_style_dashed": "虚线",
    "rl_style_dotted": "点线",
    "rl_style_dashdot": "点划线",
    "rl_style_solid": "实线",
    "rl_axis": "对应的坐标轴",
    "tb_on": "显示在哪些图上",
})
TR["ja"].update({
    "title": "電気化学 論文用グラフ作成ツール",
    "mode_label": "測定の種類",
    "upload": "データをアップロード（複数選択で重ね描き）",
    "use_demo": "ファイル未選択時はサンプルデータを使用",
    "enc": "文字コード（ヘッダーが文字化けしたときに手動で選択）",
    "enc_auto": "自動判定",
    "enc_gbk": "GBK（中国語 Windows）",
    "enc_sjis": "Shift-JIS（日本語 Windows）",
    "info_upload": "測定装置から書き出した txt / CSV / xlsx をアップロードするか、左のサンプルデータで試してください。",
    "demo_note": "表示中のものはサンプルデータです（あなたのデータではありません）。",
    "parse_fail": "{name} の読み込みに失敗しました：{err}",
    "err_two_cols": "データには 2 列以上が必要です。",
    "sec_data": "データ列",
    "grp_data": "データ",
    "grp_proc": "データ処理",
    "grp_axes": "座標軸",
    "grp_curves": "曲線と凡例",
    "grp_textbox": "テキストボックス / 注釈",
    "grp_fig": "図の外観",
    "sub_cols": "データ列",
    "sub_rows": "使用する行の範囲",
    "sub_curves": "曲線の色",
    "sub_legend": "凡例",
    "num_tb": "テキストボックスの数",
    "tb_title": "テキストボックス #{n}",
    "tb_text": "テキスト（$E_{pa}$ などの数式に対応）",
    "tb_coord": "座標の種類",
    "coord_data": "データ座標",
    "coord_rel": "相対座標（0 左/下 → 1 右/上）",
    "tb_x": "x 座標",
    "tb_y": "y 座標",
    "tb_fs": "文字サイズ / pt",
    "tb_color": "文字色",
    "tb_frame": "枠のスタイル",
    "frame_none": "枠なし（背景は透明）",
    "frame_box": "角丸の枠（半透明の背景）",
    "frame_solid": "四角い枠（白い背景）",
    "tb_rot": "回転角 / °",
    "tb_on": "表示するグラフ",
    "ax_x": "下部 x 軸",
    "ax_x2": "上部 x 軸 (x2)",
    "ax_y": "左 y 軸",
    "ax_y2": "右 y 軸 (y2)",
    "ax_color": "軸の色（ラベル・目盛り・軸線）",
    "color_xb": "下部 x 軸の色",
    "sub_ticks_common": "目盛りの設定（全軸共通）",
    "color_x2curve": "{name} 上部 x 軸の曲線の色",
    "leg_text_x2": "{name}（上部 x 軸）の凡例名",
    "unit_auto": "自動（列名から）",
    "q_seg": "電荷量の区間分け",
    "q_summary": "電荷量 / 比容量のまとめ（区間ごと）",
    "time_unit": "表示する時間の単位",
    "pot_target": "電位の基準 / 換算先",
    "pot_raw": "換算しない（参照電極基準のまま）",
    "to_she": "SHE 基準に換算",
    "to_rhe": "RHE 基準に換算",
    "custom_offset": "電位オフセット（任意） / V",
    "ref_custom": "カスタム参照電極…",
    "ref_custom_name": "参照電極の名称",
    "ref_custom_E": "この参照電極の SHE 基準の電位 / V",
    "reset_mode": "除外後の座標の詰め方",
    "reset_all": "すべて詰める（除外した隙間をすべて埋める）",
    "reset_head": "先頭だけ詰める（最初の点を 0 に、途中の隙間は残す）",
    "reset_off": "詰めない",
    "reset_note": "詰める処理は時間軸にのみ働きます。CV の電位は変更されません。",
    "sec_q_capacity": "電荷量・比容量の計算",
    "calc_q": "電荷量グラフを作成 (Charge Q)",
    "calc_cap": "比容量グラフを作成 (Specific Capacity)",
    "scan_rate": "掃引速度 v / (mV s⁻¹)",
    "active_mass": "活物質の質量（0 なら質量で規格化しない）",
    "mass_unit": "質量の単位",
    "q_fig_title": "電荷量グラフ (Charge Q)",
    "cap_fig_title": "比容量の比較グラフ (Specific Capacity)",
    "leg_manual": "位置を手動で指定",
    "leg_x": "凡例 x（0 左 → 1 右）",
    "leg_y": "凡例 y（0 下 → 1 上）",
    "leg_manual_note": "凡例の左上を基準点とし、描画領域に対する比率で指定します。0–1 の範囲外にすると図の外に置けます。",
    "leg_ncol": "凡例の列数",
    "leg_frame": "凡例の枠線を表示",
    "x2_note": "上部 x 軸：選んだ列を横軸にして、左 y 軸のデータをもう一本描きます（色と凡例名は個別に設定できます）。",
    "xcol": "x 列",
    "ycol": "y 列（左軸）",
    "start": "開始行",
    "end": "終了行（0 = 最後まで）",
    "sec_dual": "二軸の設定 (x2 / y2)",
    "dual": "右側の第 2 y 軸 (y2) を有効にする",
    "dual_x": "上側の第 2 x 軸 (x2) を有効にする",
    "y2col": "右軸のデータ列",
    "y2kind": "右軸の物理量",
    "x2col": "上部 x 軸のデータ列",
    "x2kind": "上部 x 軸の物理量",
    "color_xt": "上部 x 軸の色",
    "kind_cur": "電流（単位換算あり）",
    "kind_pot": "電位（参照電極の換算あり）",
    "kind_time": "時間（換算なし）",
    "kind_other": "その他（換算なし）",
    "sec_units": "単位と参照電極",
    "cur_raw": "元データの電流単位",
    "cur_out": "出力する電流単位",
    "norm": "電極面積で規格化して電流密度にする",
    "area": "電極面積 / cm²",
    "ref": "参照電極",
    "ref_none": "なし（参照電極を表示しない）",
    "ph": "溶液の pH",
    "sec_clean": "外れ値の除外",
    "hampel": "外れ値を自動検出 (Hampel)",
    "win": "ウィンドウ幅（点）",
    "hampel_note": "ウィンドウ幅としきい値を調整して、突発的なスパイクを除外します。",
    "nsig": "しきい値（σ）",
    "ex_range": "x の範囲で除外",
    "ex_idx": "行番号で除外",
    "sec_smooth": "平滑化",
    "sm_method": "平滑化の方法",
    "sm_none": "平滑化しない",
    "sm_ma": "移動平均",
    "sm_gauss": "ガウシアンフィルタ",
    "sm_win": "ウィンドウ幅（点）",
    "sm_poly": "多項式の次数",
    "sm_right": "右軸の曲線も平滑化する",
    "sm_raw": "元データも薄い色で表示",
    "sm_note": "フィットには平滑化前の元データを使います。",
    "sec_fit": "フィット（左軸の曲線が対象）",
    "fit_model": "モデル",
    "deg": "多項式の次数",
    "fit_range": "フィット範囲",
    "fit_none": "フィットしない",
    "fit_linear": "直線",
    "fit_poly": "多項式",
    "fit_exp": "指数減衰",
    "sec_fig": "図",
    "width": "幅",
    "w85": "1 段組 85 mm",
    "w120": "1.5 段組 120 mm",
    "w175": "2 段組 175 mm",
    "fontsize": "文字サイズ / pt",
    "scatter": "データを点で表示",
    "legend": "凡例を表示",
    "legend_loc": "凡例の位置",
    "sec_range": "座標軸の範囲（空欄なら自動）",
    "xmin": "x の最小値",
    "xmax": "x の最大値",
    "ymin": "y の最小値",
    "ymax": "y の最大値",
    "y2min": "右 y の最小値",
    "y2max": "右 y の最大値",
    "x2min": "上部 x の最小値",
    "x2max": "上部 x の最大値",
    "warn_num": "座標軸の範囲と目盛り間隔には数値を入力してください。",
    "step_title": "主目盛りの間隔",
    "stepx": "x の間隔",
    "stepy": "y の間隔",
    "stepy2": "右 y の間隔",
    "stepx2": "上部 x の間隔",
    "warn_step": "主目盛りの間隔が小さすぎるため無視しました。",
    "lab_note": "LaTeX の数式記法に対応しています（例：`$E_{pa}$`）。",
    "sec_ticks": "目盛りの表示",
    "tick_marks": "目盛り線",
    "tick_nums": "目盛りの数値",
    "minor": "補助目盛りを表示",
    "tdir": "目盛りの向き",
    "dir_in": "内向き",
    "dir_out": "外向き",
    "sec_color": "色",
    "color_curve": "{name} の曲線の色",
    "color_left": "{name} 左軸の曲線の色",
    "color_right": "{name} 右軸の曲線の色",
    "color_fit": "フィット線の色",
    "yaxis_color": "y 軸の色",
    "yc_follow": "曲線の色に合わせる",
    "color_yl": "左 y 軸の色",
    "color_yr": "右 y 軸の色",
    "sec_leg_text": "凡例名のカスタマイズ（空欄なら既定の名前）",
    "leg_text_single": "{name} の凡例名",
    "leg_text_left": "{name}（左軸）の凡例名",
    "leg_text_right": "{name}（右軸）の凡例名",
    "sec_labels": "軸ラベル",
    "lab_x": "x 軸ラベル",
    "lab_x2": "上部 x 軸ラベル",
    "lab_y": "左 y 軸ラベル",
    "lab_y2": "右 y 軸ラベル",
    "lab_help": "LaTeX の数式に対応しています。",
    "sec_ai": "AI 分析",
    "ai_intro": "処理後のデータの要約を Gemini に送って、解析を補助します。",
    "ai_warn": "参考情報です。結果は必ずご自身で確認してください。",
    "ai_key": "API キー",
    "ai_key_help": "メモリ上にのみ保持します。",
    "ai_model": "モデル",
    "ai_model_help": "Google Gemini のモデル名。",
    "ai_ctx": "実験の背景（任意）",
    "ai_ctx_ph": "例：0.1 M PBS、グルコースの検出",
    "ai_scan": "掃引速度 / mV s⁻¹",
    "ai_q": "質問（任意）",
    "ai_sendcurve": "間引いたデータも送信する",
    "ai_preview": "AI に送る内容を確認",
    "ai_btn": "AI 分析を実行",
    "ai_spin": "AI が分析中…",
    "ai_nokey": "API キーを入力してください。",
    "ai_nopkg": "google-genai パッケージがインストールされていません。",
    "ai_fail": "AI の呼び出しに失敗しました：{err}",
    "ai_stale": "（設定が変更されました。もう一度実行してください。）",
    "sec_peaks": "CV のピーク解析",
    "pk_note": "ベースライン補正は行っていません。",
    "pk_none": "明確なピークは検出されませんでした。",
    "tab_static": "静的プレビュー",
    "tab_inter": "インタラクティブ表示",
    "tab_table": "処理後のデータ",
    "hover_mode": "ホバー方式",
    "hv_x": "同一 x 軸",
    "crosshair": "十字カーソル",
    "show_removed": "除外した点を表示（赤い ×）",
    "inter_hint": "カーソルを合わせると数値を確認できます。",
    "no_plotly": "plotly がインストールされていません。",
    "show_peaks": "ピーク位置を表示",
    "tbl_only_kept": "除外していない点のみ",
    "tbl_file_sel": "ファイルを選択",
    "tbl_dl": "処理後のデータをダウンロード (CSV)",
    "tbl_hint": "処理後のデータは Origin にそのまま読み込めます。",
    "warn_range": "範囲の書式が正しくありません。",
    "preview_caption": "プレビュー",
    "results": "結果",
    "tbl_file": "ファイル",
    "tbl_removed": "除外した点の数",
    "tbl_params": "パラメータ",
    "fit_fail": "フィットに失敗しました：{err}",
    "sec_export": "画像の出力と保存",
    "fmt": "形式",
    "dpi": "解像度 (dpi)",
    "fname": "ファイル名の接頭辞",
    "dl": "主図をダウンロード",
    "dl_q": "電荷量グラフをダウンロード",
    "dl_cap": "比容量グラフをダウンロード",
    "dl_hint": "ベクター形式 (PDF / SVG) またはビットマップ形式 (PNG / TIFF)。",
    "fig_title": "グラフのタイトル（任意）",
    "follow_main": "曲線の色を主図に合わせる",
    "fig_main": "主図",
    "fig_q": "電荷量グラフ",
    "fig_cap": "比容量グラフ",
    "grp_qfig": "電荷量グラフの設定",
    "grp_capfig": "比容量グラフの設定",
    "grp_reflines": "基準線（破線など）",
    "rl_n": "基準線の数",
    "rl_title": "基準線 #{n}",
    "rl_orient": "向き",
    "rl_orient_h": "水平線（y = 一定値）",
    "rl_orient_v": "垂直線（x = 一定値）",
    "rl_pos": "位置（データ座標）",
    "rl_color": "線の色",
    "rl_style": "線の種類",
    "rl_style_dashed": "破線",
    "rl_style_dotted": "点線",
    "rl_style_dashdot": "一点鎖線",
    "rl_style_solid": "実線",
    "rl_axis": "対応する軸",
})


def t(key, **kw):
    s = TR.get(lang, {}).get(key) or TR.get("zh", {}).get(key, key)
    return s.format(**kw) if kw else s


# ---------------------------------------------------------------- 数据解析
import csv

TSEC = {"s": 1.0, "ms": 1e-3, "min": 60.0, "h": 3600.0}  # 换算成秒的系数


def decode_bytes(raw: bytes, enc: str, pref: str) -> str:
    if enc != "auto":
        return raw.decode(enc, errors="replace")
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        pass
    try:
        return raw.decode("gb18030")
    except UnicodeDecodeError:
        pass
    try:
        return raw.decode("cp932")
    except UnicodeDecodeError:
        pass
    return raw.decode("latin-1")


def _split_line(line, delim):
    if delim == "ws":
        return line.split()
    return [s.strip() for s in line.rstrip("\r\n").split(delim)]


def _numeric_row(tokens, dec, tol=False):
    # 全是数字（空单元格按 NaN）且至少 2 个数字才算数据行；
    # tol=True（宽松模式）时允许夹杂文字列（如 “工步类型”），文字按 NaN，但文字不能比数字多
    vals, n_num, n_txt = [], 0, 0
    for s in tokens:
        if s == "":
            vals.append(np.nan)
            continue
        try:
            vals.append(float(s.replace(",", ".") if dec else s))
            n_num += 1
        except ValueError:
            if not tol:
                return None
            vals.append(np.nan)
            n_txt += 1
    return vals if n_num >= 2 and n_txt <= n_num else None


def _detect_format(lines, tol=False):
    # 找到第一行数据，推断分隔符（制表符 / 分号 / 逗号 / 空白）和小数点（点 / 逗号）
    for line in lines:
        if not line.strip():
            continue
        cands = []
        if "\t" in line:
            cands += [("\t", False), ("\t", True)]
        if ";" in line:
            cands += [(";", False), (";", True)]
        if "," in line:
            cands += [(",", False), ("ws", True)]
        cands.append(("ws", False))
        for delim, dec in cands:
            if _numeric_row(_split_line(line, delim), dec, tol) is not None:
                return delim, dec
    return None


def _trim_tail(tokens):
    tokens = list(tokens)
    while tokens and tokens[-1] == "":
        tokens.pop()
    return tokens


def _header_tokens(line, delim, n, lead_empty=False):
    # 表头行拆成 n 列；支持带引号的 CSV，以及用两个以上空格分隔、列名里带空格的表头
    if delim == "ws":
        for toks in (re.split(r"\t|\s{2,}", line.strip()), line.split()):
            if len(toks) == n:
                return [s.strip() for s in toks]
        return None
    try:
        toks = next(csv.reader([line], delimiter=delim))
    except Exception:
        toks = line.split(delim)
    toks = _trim_tail([s.strip() for s in toks])
    if lead_empty and len(toks) == n - 1:  # 数据行以分隔符开头（Gamry），表头没写开头的分隔符
        toks = [""] + toks
    return toks if len(toks) == n else None


def _dedupe(names):
    seen, out = {}, []
    for nm in names:
        k = seen.get(nm, 0)
        seen[nm] = k + 1
        out.append(nm if k == 0 else f"{nm}.{k}")
    return out


def _find_names(lines, start, n, delim, dec, lead_empty=False, tol=False):
    # 数据块上方最近的非空行当表头；如果上面紧挨着两行都是 n 列的文字行（Gamry：名称行 + 单位行），合并成 名称/单位
    found = []
    j = start - 1
    while j >= 0 and len(found) < 2:
        if lines[j].strip():
            toks = _header_tokens(lines[j], delim, n, lead_empty)
            if toks is None or _numeric_row(toks, dec, tol) is not None:
                break
            found.append(toks)
        j -= 1
    if not found:
        return None
    if len(found) == 2:
        units, names = found[0], found[1]
        if all(len(u) <= 12 for u in units) and any(len(s) >= 2 for s in names):
            return [f"{a}/{b}" if a and b and b != "#" else (a or b) for a, b in zip(names, units)]
    return found[0]


def _parse_text_mode(text, tol):
    lines = text.splitlines()
    fmt = _detect_format(lines, tol)
    if fmt is None:
        raise ValueError("no numeric data")
    delim, dec = fmt
    explicit = delim != "ws"
    blocks, skipped, i, N = [], 0, 0, len(lines)
    while i < N:
        first = _split_line(lines[i], delim) if lines[i].strip() else None
        if first is None or _numeric_row(first, dec, tol) is None:
            i += 1
            continue
        start, raws = i, []
        while i < N:
            if not lines[i].strip():
                i += 1
                continue
            toks = _split_line(lines[i], delim)
            if _numeric_row(toks, dec, tol) is None:
                break
            raws.append(toks)
            i += 1
        counts = [len(_trim_tail(r)) if explicit else len(r) for r in raws[:30]]
        n = max(set(counts), key=counts.count)
        rows = []
        for toks in raws:
            toks = _trim_tail(toks) if explicit else toks
            if len(toks) > n or (len(toks) < n and not explicit):
                skipped += 1
                continue
            toks = toks + [""] * (n - len(toks))
            rows.append(_numeric_row(toks, dec, tol))
        if rows:
            lead_empty = explicit and all(r and r[0] == "" for r in raws)
            blocks.append([_find_names(lines, start, n, delim, dec, lead_empty, tol), rows, n])
    if not blocks:
        raise ValueError("no numeric data")
    # 没有自己表头的数据块（比如只有一行 “Cycle 2” 标题），沿用前面同列数数据块的列名
    for k, b in enumerate(blocks):
        if b[0] is None:
            prev = [c[0] for c in blocks[:k] if c[0] is not None and c[2] == b[2]]
            b[0] = prev[-1] if prev else [f"col{j}" for j in range(b[2])]
    groups = {}
    for names, rows, n in blocks:
        groups.setdefault(tuple(names), []).append(rows)
    names, parts = max(groups.items(), key=lambda kv: sum(len(r) for r in kv[1]))
    data = [r for part in parts for r in part]
    df = pd.DataFrame(data, columns=_dedupe([s if s else f"col{k}" for k, s in enumerate(names)]))
    df = df.dropna(axis=1, how="all")
    if df.shape[1] < 2:
        raise ValueError("no numeric data")
    df.attrs["skipped_rows"] = skipped
    return df


def _parse_text(text):
    # 先严格解析；失败，或只得到很少几行（可能只是文件头里的几行数字）时，再用宽松模式试一次，取较大的结果
    best = None
    for tol in (False, True):
        try:
            df = _parse_text_mode(text, tol)
        except ValueError:
            continue
        if best is None or df.size > best.size:
            best = df
        if not tol and df.shape[0] >= 10:
            break
    if best is None:
        raise ValueError("no numeric data")
    return best


def parse_table(raw: bytes, enc: str = "auto", pref: str = "zh") -> pd.DataFrame:
    return _parse_text(decode_bytes(raw, enc, pref))


_TIME_STR = re.compile(r"^\s*(?:(\d+):)?(\d{1,2}):(\d{2}(?:\.\d+)?)\s*$")


def _excel_sheet_to_text(sheet):
    # 把 Excel 里的 “0:00:30” 文本时间、日期时间换成秒，再转成制表符文本，交给同一个解析器
    sheet = sheet.copy()
    for c in sheet.columns:
        col = sheet[c]
        conv, first = {}, None
        for r, v in col.items():
            if hasattr(v, "to_pydatetime") or type(v).__name__ in ("datetime", "Timestamp"):
                first = v if first is None else first
                conv[r] = (v - first).total_seconds()
            elif isinstance(v, str):
                m = _TIME_STR.match(v)
                if m:
                    conv[r] = int(m.group(1) or 0) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
            elif type(v).__name__ == "time":
                conv[r] = v.hour * 3600 + v.minute * 60 + v.second + v.microsecond / 1e6
        if len(conv) >= 2:
            rmin = min(conv)
            for r, v in conv.items():
                sheet.at[r, c] = v
            pos = list(col.index).index(rmin)
            if pos > 0:
                head = sheet.iat[pos - 1, list(sheet.columns).index(c)]
                if isinstance(head, str) and head.strip():
                    sheet.iat[pos - 1, list(sheet.columns).index(c)] = head.strip() + " (s)"
    return sheet.to_csv(sep="\t", header=False, index=False, na_rep="")


@st.cache_data
def load_file(name: str, raw: bytes, enc: str, pref: str) -> pd.DataFrame:
    if name.lower().endswith(".xlsx"):
        sheets = pd.read_excel(io.BytesIO(raw), sheet_name=None, header=None, dtype=object)
        best = None
        for sh in sheets.values():
            try:
                df = _parse_text(_excel_sheet_to_text(sh))
            except ValueError:
                continue
            if best is None or df.size > best.size:
                best = df
        if best is None:
            raise ValueError("no numeric data")
        return best
    return parse_table(raw, enc, pref)


# ---------------------------------------------------------------- 列角色与单位识别
def _norm(name):
    return str(name).strip().lower().replace("µ", "u").replace("μ", "u")


def _kw(*words):
    return "|".join(rf"(?<![a-z]){w}(?![a-z])" for w in words)


_R_TIME = re.compile(_kw("time", "elapsed", "duration", "t") + r"|时间|時間|経過|耗时|秒")
_R_CUR = re.compile(_kw("current", "curr", "im", "i") + r"|<i>|电流|電流")
_R_CHG = re.compile(r"capacity|容量|q-qo|电量|電荷|mah|charge|\bq\b")
_R_POT = re.compile(_kw("potential", "voltage", "volt", "ewe", "ecell", "vf", "vm", "vu", "e", "u") + r"|<e>|电压|电位|电势|電圧|電位")
_R_IDX = re.compile(_kw("mode", "index", "cycle", "loop", "step", "status", "state", "record", "number", "error") + r"|^#|^pt(?![a-z])|ox/red|序号|序號|循环|循環|工步|状态|状態|记录|点数")
_BASE_SINGLE = {"t": "time", "i": "cur", "im": "cur", "e": "pot", "u": "pot", "v": "pot", "vf": "pot", "vm": "pot"}


def col_info(name):
    n = _norm(name)
    base = re.split(r"[/(\[\s]", n)[0]
    pen, role = 0, "other"
    if _R_TIME.search(n) or _BASE_SINGLE.get(base) == "time":
        role = "time"
        pen += 10 if re.search(r"step|工步|stage|阶段", n) else 0
    elif re.search(r"density|密度|/cm", n):
        role = "other"
    elif _R_CUR.search(n) or _BASE_SINGLE.get(base) == "cur":
        role = "cur"
        pen += 5 if re.search(r"control|ctrl|setpoint|applied|设定", n) else 0
    elif _R_CHG.search(n):
        role = "charge"
    elif _R_POT.search(n) or _BASE_SINGLE.get(base) == "pot":
        role = "pot"
        pen += 5 if re.search(r"control|ctrl|setpoint|applied|eapp|设定", n) else 0
        pen += 20 if re.search(r"(?<![a-z])ece(?![a-z])|counter|对电极", n) else 0
        pen += 2 if re.search(r"(?<![a-z])vu(?![a-z])", n) else 0
    elif _R_IDX.search(n):
        role = "idx"
    return {"role": role, "pen": pen}


_UNIT_WORDS = {
    "time": {"s": "s", "sec": "s", "secs": "s", "second": "s", "seconds": "s", "秒": "s", "ms": "ms",
             "min": "min", "mins": "min", "minute": "min", "minutes": "min", "分": "min", "分钟": "min",
             "h": "h", "hr": "h", "hrs": "h", "hour": "h", "hours": "h", "时": "h", "小时": "h", "時間": "h"},
    "cur": {"a": "A", "ma": "mA", "ua": "µA", "na": "nA", "pa": "pA"},
    "pot": {"v": "V", "mv": "mV"},
}


def parse_unit(name, use):
    # 从列名里读单位：Time/s、电流(mA)、<I>/mA、测试时间[h]、V vs. Ref. …；没写就返回 None
    n = _norm(name)
    words = sorted(_UNIT_WORDS[use], key=len, reverse=True)
    pat = r"(?:/|\(|\[|（|【|\s|\.)(" + "|".join(re.escape(w) for w in words) + r")(?:\)|\]|）|】|\s|\.|/|$)"
    m = re.search(pat, n)
    return _UNIT_WORDS[use][m.group(1)] if m else None


_VALID_UNITS = {"time": ("s", "ms", "min", "h"), "cur": ("A", "mA", "µA", "nA", "pA"), "pot": ("V", "mV")}


def eff_unit(setting, name, use, default):
    # 手动选了合法单位就用它；“auto”或任何不认识的值都按列名识别，列名没写单位就用默认值
    if setting in _VALID_UNITS[use]:
        return setting
    return parse_unit(name, use) or default


def pick_col(infos, role, exclude=()):
    best = None
    for i, c in enumerate(infos):
        if i in exclude or c["role"] != role:
            continue
        key = (c["pen"], i)
        if best is None or key < best[0]:
            best = (key, i)
    return None if best is None else best[1]


WANT_ROLES = {"CV": ("pot", "cur"), "CA": ("time", "cur"), "CP": ("time", "pot")}


def default_cols(mode, infos):
    used = set()
    out = []
    for role in WANT_ROLES[mode]:
        i = pick_col(infos, role, used)
        if i is not None:
            used.add(i)
        out.append(i)
    for k, i in enumerate(out):  # 认不出来的角色，退回到还没用过的第一列
        if i is None:
            i = next((j for j in range(len(infos)) if j not in used), 0)
            used.add(i)
            out[k] = i
    return out[0], out[1]


def resolve_col(df, infos_f, name_sel, role_sel, idx_fallback):
    # 多个文件列顺序不同时：先按列名找，找不到再按角色找，最后才按位置
    cols_f = list(df.columns)
    if name_sel in cols_f:
        return cols_f.index(name_sel)
    if role_sel in ("time", "pot", "cur", "charge"):
        i = pick_col(infos_f, role_sel)
        if i is not None:
            return i
    return idx_fallback if idx_fallback < len(cols_f) else 0


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
        t_arr = np.linspace(0.05, 60, 600)
        i = 2e-6 / np.sqrt(t_arr) + 1e-7 + rng.normal(0, 2e-8, t_arr.size)
        i[[80, 300]] += 8e-7
        return pd.DataFrame({"Time/sec": t_arr, "Current/A": i})
    t_arr = np.linspace(0, 100, 500)
    e = 0.2 + 0.25 * (1 - np.exp(-t_arr / 20)) + rng.normal(0, 0.002, t_arr.size)
    e[[100, 350]] += 0.05
    return pd.DataFrame({"Time/sec": t_arr, "Potential/V": e, "Current/mA": np.full_like(t_arr, 0.5)})


# ---------------------------------------------------------------- 电量与比容量
def _sign_runs(sig, min_len=3):
    # 把逐点的方向（-1/0/+1）整理成连续的区段：0 沿用前一个方向，短于 min_len 的小段并入前一段
    s = np.asarray(sig, dtype=int).copy()
    m = s.size
    nz = np.flatnonzero(s)
    if nz.size == 0:
        return np.zeros(m, dtype=int)
    s[: nz[0]] = s[nz[0]]
    last = np.maximum.accumulate(np.where(s != 0, np.arange(m), 0))
    s = s[last]
    b = np.flatnonzero(np.diff(s) != 0) + 1
    starts, ends = np.r_[0, b], np.r_[b, m]
    out = []
    for a, e in zip(starts, ends):
        sg = int(s[a])
        if out and (e - a < min_len or out[-1][0] == sg):
            out[-1][2] = e
        else:
            out.append([sg, a, e])
    res = np.zeros(m, dtype=int)
    for sg, a, e in out:
        res[a:e] = sg
    return res


def compute_charge(mode, t_sec, i_amp, e_volt, keep, abs_q, seg_auto, mass_g, scan_v):
    # t_sec: 秒；i_amp: 安培；e_volt: 伏特（仅 CV）；keep: 保留点掩码。
    # 只用保留下来的点做梯形积分；时间不增加的地方不积分；seg_auto 时按电流正负号（CV 按扫描方向）分段，每段从 0 累计。
    n = keep.size
    k = np.flatnonzero(keep)
    if k.size < 2:
        raise ValueError("few")
    i = i_amp[k]
    if mode == "CV":
        e = e_volt[k]
        de = np.diff(e, prepend=e[0])
        if t_sec is not None:
            tt = t_sec[k]
        else:
            if not scan_v or scan_v <= 0:
                raise ValueError("no_time")
            tt = np.cumsum(np.abs(de)) / scan_v
        sig = np.where(de > 1e-12, 1, np.where(de < -1e-12, -1, 0))
    else:
        if t_sec is None:
            raise ValueError("no_time")
        tt = t_sec[k]
        eps = 0.02 * np.nanmax(np.abs(i)) if np.isfinite(i).any() else 0.0
        sig = np.where(i > eps, 1, np.where(i < -eps, -1, 0))
    dt = np.diff(tt, prepend=tt[0])
    valid = dt > 0
    valid[0] = False
    if seg_auto:
        run = _sign_runs(sig)
        bound = np.zeros(k.size, dtype=bool)
        bound[0] = True
        bound[1:] = (run[1:] != run[:-1]) | (dt[1:] < 0)  # 时间倒退才算新分段；时间相同（顶点重复点）只是不积分
        seg = np.cumsum(bound) - 1
        valid &= ~bound
        dirn = run
    else:
        seg = np.zeros(k.size, dtype=int)
        dirn = np.zeros(k.size, dtype=int)
    f = np.abs(i) if abs_q else i
    dq = np.zeros(k.size)
    dq[1:] = 0.5 * (f[1:] + f[:-1]) * dt[1:]
    dq = np.where(valid, dq, 0.0)
    c = np.cumsum(dq)
    starts = np.flatnonzero(np.r_[True, seg[1:] != seg[:-1]])
    q = c - c[starts[seg]]
    cap = q / 3.6  # 1 mAh = 3.6 C
    if mass_g and mass_g > 0:
        cap = cap / mass_g
        cap_unit = "mAh g$^{-1}$"
    else:
        cap_unit = "mAh"
    q_full, cap_full = np.full(n, np.nan), np.full(n, np.nan)
    seg_full, dir_full = np.full(n, np.nan), np.full(n, np.nan)
    q_full[k], cap_full[k], seg_full[k], dir_full[k] = q, cap, seg, dirn
    segs = []
    for s in np.unique(seg):
        m = np.flatnonzero(seg == s)
        segs.append({"seg": int(s) + 1, "dir": int(dirn[m[0]]), "n": int(m.size), "q": float(q[m[-1]]), "cap": float(cap[m[-1]]),
                     "dur_s": float(tt[m[-1]] - tt[m[0]])})
    return {"q_coulombs": q_full, "capacity": cap_full, "q_seg": seg_full, "q_dir": dir_full, "cap_unit": cap_unit, "q_segs": segs}


# ---------------------------------------------------------------- 数据处理
def cscale(p):
    area = p["area"] if p["norm"] else 1.0
    return CUR[p["cur_raw"]] / CUR[p["cur_out"]] / area


def transform(x, y, mode, p):
    ps = p.get("pot_scale", 1.0)
    if mode == "CV":
        return x * ps + p["shift"], y * cscale(p)
    if mode == "CA":
        return x, y * cscale(p)
    return x, y * ps + p["shift"]


def transform_y2(y2, kind, p):
    if kind == "cur":
        return y2 * cscale(p)
    if kind == "pot":
        return y2 * p.get("pot_scale", 1.0) + p["shift"]
    return y2


def transform_x2(x2, kind, p):
    if kind == "pot":
        return x2 * p.get("pot_scale", 1.0) + p["shift"]
    if kind == "time":
        return time_conv(x2 * p.get("t_scale", 1.0), p.get("time_unit", "s"))
    return x2


def renumber_x(x, keep):
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
    x = x.copy()
    first = np.where(keep)[0][0]
    return x - x[first]


def time_conv(x, unit):
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
    s = str(s).strip()
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
    "produced by a plotting tool; the numbers were computed by the tool. Base every statement on the provided numbers "
    "and never invent values. Separate clearly: (1) what the numbers show, (2) possible interpretations, (3) next steps. "
    "Reply in {language}."
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
            "font.family": FONTS_BY_LANG.get(lang, FONTS),
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
    target = p.get("pot_target", "raw")
    if target == "she":
        return "Potential / V vs. SHE"
    if target == "rhe":
        return "Potential / V vs. RHE"
    ref = p["ref"]
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


# ---------------------------------------------------------------- 绘图生成器
# ---------------------------------------------------------------- 各图共用：参考线、文本框、图例、范围与间隔
_LS = {"dashed": "--", "dotted": ":", "dashdot": "-.", "solid": "-"}
_PLOTLY_DASH = {"dashed": "dash", "dotted": "dot", "dashdot": "dashdot", "solid": "solid"}


def _tb_for(p, which):
    # 文本框可以分别选择显示在主图 / 电量图 / 比容量图上（旧数据没有 "on" 时三张图都显示）
    return [tb for tb in (p.get("textboxes") or []) if which in tb.get("on", ("main", "q", "cap"))]


def _draw_reflines(a_x, a_y, lines, a_x2=None, a_y2=None):
    # 参考线：水平 / 竖直、任意位置、颜色和线型；有双轴时可以指定对应哪条轴
    for ln in lines or []:
        kw = dict(color=ln["color"], ls=_LS.get(ln["style"], "--"), lw=1.0, zorder=1)
        if ln["orient"] == "h":
            (a_y2 if (ln.get("axis") == "y2" and a_y2 is not None) else a_y).axhline(ln["pos"], **kw)
        else:
            (a_x2 if (ln.get("axis") == "x2" and a_x2 is not None) else a_x).axvline(ln["pos"], **kw)


def _draw_text(ax, tb):
    txt = cjk_safe(tb["text"])
    kwargs = {"fontsize": tb["fs"], "color": tb["color"], "rotation": tb["rot"], "ha": "center", "va": "center"}
    if tb["frame"] == "box":
        kwargs["bbox"] = dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#888888", linewidth=0.8, alpha=0.85)
    elif tb["frame"] == "solid":
        kwargs["bbox"] = dict(boxstyle="square,pad=0.3", facecolor="white", edgecolor="#000000", linewidth=0.8, alpha=1.0)
    if tb["coord"] == "rel":
        ax.text(tb["x"], tb["y"], txt, transform=ax.transAxes, **kwargs)
    else:
        ax.text(tb["x"], tb["y"], txt, **kwargs)


def _make_legend(a, handles, labels, p):
    kw = dict(ncol=p.get("legend_ncol", 1), frameon=p.get("legend_frame", False))
    if p.get("legend_loc") == "manual":
        kw.update(loc="upper left", bbox_to_anchor=p["legend_xy"])
    else:
        kw.update(loc=p.get("legend_loc", "best"))
    if kw["frameon"]:
        kw.update(edgecolor="black", fancybox=False, framealpha=1.0)
    leg = a.legend(handles, labels, **kw)
    if kw["frameon"]:
        leg.get_frame().set_linewidth(0.8)
    return leg


def _limits_steps(ax, p):
    if p.get("xlim") and p["xlim"] != (None, None):
        ax.set_xlim(*p["xlim"])
    if p.get("ylim") and p["ylim"] != (None, None):
        ax.set_ylim(*p["ylim"])
    skipped = False
    for axis, step in (("x", p.get("xstep")), ("y", p.get("ystep"))):
        if step is None:
            continue
        lo, hi = ax.get_xlim() if axis == "x" else ax.get_ylim()
        if abs(hi - lo) / step > 200:
            skipped = True
            continue
        (ax.xaxis if axis == "x" else ax.yaxis).set_major_locator(MultipleLocator(step))
    return skipped


def _plot_segments(ax, xs, ys, r, color, label, scatter=False):
    # 每个分段单独画一条线（充电、放电之间不连线）；返回第一条线用于图例
    keep = r["keep"]
    xs, ys = xs[keep], ys[keep]
    seg = r.get("q_seg")
    masks = [np.ones(xs.size, dtype=bool)] if seg is None else [seg[keep] == s for s in np.unique(seg[keep][np.isfinite(seg[keep])])]
    first = None
    for m in masks:
        args = (xs[m], ys[m], "o") if scatter else (xs[m], ys[m])
        kw = dict(color=color, label=label if first is None else None)
        if scatter:
            kw["ms"] = 2
        (h,) = ax.plot(*args, **kw)
        first = first or h
    return first


def build_q_fig(results, mode, p):
    """电量图：Q 对 电位（CV）或时间（CA / CP）。设置项和主图一致，由 p 里的同名键提供。"""
    set_style(p)
    fig, ax = plt.subplots(figsize=(p["w"] * MM, p["h"] * MM), constrained_layout=True)
    handles, labels = [], []
    custom = p.get("custom_legends", [])
    for k, r in enumerate(results):
        if "q_coulombs" not in r:
            continue
        name = custom[k][0] if k < len(custom) and custom[k][0] else r["name"]
        h = _plot_segments(ax, r["x"], r["q_coulombs"], r, p["colors"][k][0], name, p.get("scatter", False))
        if h is not None:
            handles.append(h)
            labels.append(name)
    xl, _ = axis_labels(mode, p)
    ax.set_xlabel(cjk_safe(p.get("lab_x") or xl))
    ax.set_ylabel(cjk_safe(p.get("lab_y") or "Charge $Q$ / C"))
    if p.get("title"):
        ax.set_title(cjk_safe(p["title"]))
    _draw_reflines(ax, ax, p.get("reflines"))
    for tb in _tb_for(p, "q"):
        _draw_text(ax, tb)
    fig._tick_skipped = _limits_steps(ax, p)
    apply_ticks(ax, None, None, p)
    color_xaxis(ax, "bottom", p["xb_col"])
    color_yaxis(ax, "left", p["yl_col"])
    if p.get("legend") and handles:
        _make_legend(ax, handles, labels, p)
    return fig


def build_cap_fig(results, mode, p):
    """比容量图：电位（或电流）对 容量。设置项和主图一致，由 p 里的同名键提供。"""
    set_style(p)
    fig, ax = plt.subplots(figsize=(p["w"] * MM, p["h"] * MM), constrained_layout=True)
    handles, labels = [], []
    cap_unit = results[0].get("cap_unit", "mAh") if results else "mAh"
    x_cap_label = f"Specific Capacity / ({cap_unit})" if "g" in cap_unit else "Capacity / mAh"
    custom = p.get("custom_legends", [])
    for k, r in enumerate(results):
        if "capacity" not in r:
            continue
        name = custom[k][0] if k < len(custom) and custom[k][0] else r["name"]
        h = _plot_segments(ax, r["capacity"], r["x"] if mode == "CV" else r["y"], r, p["colors"][k][0], name, p.get("scatter", False))
        if h is not None:
            handles.append(h)
            labels.append(name)
    auto_y = pot_label(p) if mode in ("CV", "CP") else cur_label(p)
    ax.set_xlabel(cjk_safe(p.get("lab_x") or x_cap_label))
    ax.set_ylabel(cjk_safe(p.get("lab_y") or auto_y))
    if p.get("title"):
        ax.set_title(cjk_safe(p["title"]))
    _draw_reflines(ax, ax, p.get("reflines"))
    for tb in _tb_for(p, "cap"):
        _draw_text(ax, tb)
    fig._tick_skipped = _limits_steps(ax, p)
    apply_ticks(ax, None, None, p)
    color_xaxis(ax, "bottom", p["xb_col"])
    color_yaxis(ax, "left", p["yl_col"])
    if p.get("legend") and handles:
        _make_legend(ax, handles, labels, p)
    return fig


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

    _draw_reflines(ax, ax, p.get("reflines"), a_x2=ax3, a_y2=ax2)
    if p.get("textboxes"):
        for tb in _tb_for(p, "main"):
            txt = cjk_safe(tb["text"])
            kwargs = {"fontsize": tb["fs"], "color": tb["color"], "rotation": tb["rot"], "ha": "center", "va": "center"}
            if tb["frame"] == "box":
                kwargs["bbox"] = dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#888888", linewidth=0.8, alpha=0.85)
            elif tb["frame"] == "solid":
                kwargs["bbox"] = dict(boxstyle="square,pad=0.3", facecolor="white", edgecolor="#000000", linewidth=0.8, alpha=1.0)

            if tb["coord"] == "rel":
                ax.text(tb["x"], tb["y"], txt, transform=ax.transAxes, **kwargs)
            else:
                ax.text(tb["x"], tb["y"], txt, **kwargs)

    ax.set_xlabel(cjk_safe(p["lab_x"] or xl))
    ax.set_ylabel(cjk_safe(p["lab_y"] or yl))
    if p.get("title"):
        ax.set_title(cjk_safe(p["title"]))

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
PLOTLY_FONTS = {"ja": PLOTLY_FONT, "zh": "Times New Roman, SimSun, Songti SC, Noto Serif CJK SC, serif"}


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

    if p.get("textboxes"):
        for tb in _tb_for(p, "main"):
            txt = html_label(tb["text"])
            is_rel = tb["coord"] == "rel"
            b_kw = {}
            if tb["frame"] == "box":
                b_kw = dict(bordercolor="#888888", borderwidth=1, bgcolor="rgba(255, 255, 255, 0.85)")
            elif tb["frame"] == "solid":
                b_kw = dict(bordercolor="#000000", borderwidth=1, bgcolor="#ffffff")
            
            fig.add_annotation(
                x=tb["x"], y=tb["y"],
                xref="paper" if is_rel else "x",
                yref="paper" if is_rel else "y",
                text=txt,
                showarrow=False,
                font=dict(size=tb["fs"], color=tb["color"]),
                textangle=-tb["rot"],
                xanchor="center",
                yanchor="middle",
                **b_kw
            )

    ticks = "inside" if p["tick_dir"] == "in" else "outside"
    spike = dict(showspikes=spikes, spikemode="across", spikesnap="data", spikethickness=1, spikedash="dot", spikecolor="#666")
    fig.update_layout(
        template="simple_white", height=470,
        margin=dict(l=70, r=70 if p["dual"] else 20, t=60 if p["dual_x"] else 20, b=60),
        font=dict(family=PLOTLY_FONTS.get(lang, PLOTLY_FONT), size=13), hovermode="closest" if hover_mode == "closest" else "x unified",
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
    for ln in p.get("reflines") or []:
        line_kw = dict(color=ln["color"], dash=_PLOTLY_DASH.get(ln["style"], "dash"), width=1.2)
        if ln["orient"] == "h":
            fig.add_shape(type="line", xref="paper", x0=0, x1=1, yref="y2" if (ln.get("axis") == "y2" and p["dual"]) else "y",
                          y0=ln["pos"], y1=ln["pos"], line=line_kw, layer="below")
        else:
            fig.add_shape(type="line", xref="x2" if (ln.get("axis") == "x2" and p["dual_x"]) else "x", x0=ln["pos"], x1=ln["pos"],
                          yref="paper", y0=0, y1=1, line=line_kw, layer="below")
    if p.get("title"):
        fig.update_layout(title=dict(text=html_label(p["title"]), x=0.5), margin=dict(t=95 if p["dual_x"] else 60))
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
    if "q_coulombs" in r:
        data["Charge / C"] = r["q_coulombs"]
    if "capacity" in r:
        cap_header = f"Capacity / ({r['cap_unit']})"
        data[cap_header] = r["capacity"]
    if "q_seg" in r:
        data["Segment"] = r["q_seg"] + 1
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
        _sk = dfs[f.name].attrs.get("skipped_rows", 0)
        if _sk:
            st.caption(t("skip_rows", name=f.name, n=_sk))
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
infos = [col_info(c) for c in cols]
dx_def, dy_def = default_cols(mode, infos)
sig = f"{mode}_{len(cols)}_{hashlib.md5('|'.join(map(str, cols)).encode('utf-8')).hexdigest()[:6]}"


def _role_text(i):
    role = infos[i]["role"]
    u = parse_unit(cols[i], role) if role in ("time", "cur", "pot") else None
    return f"{cols[i]} → {t('role_' + role)}" + (f" ({u})" if u else "")


def _auto_fmt(u):
    return t("unit_auto") if u == "auto" else u


def _safe_select(label, options, key, fmt=None, index=0, where=None):
    # 个别 Streamlit 版本在切换语言 / 沿用旧状态时，会把“显示出来的文字”当成取值交回来；这里把它换回真正的选项
    fmt = fmt or (lambda o: o)
    v = (where or st).selectbox(label, options, index=index, format_func=fmt, key=key)
    if v in options:
        return v
    global lang
    cur = lang
    try:
        for lg in ("zh", "ja"):
            lang = lg
            back = {fmt(o): o for o in options}
            if v in back:
                return back[v]
    finally:
        lang = cur
    return options[index]


def _detected_units_text():
    parts = []

    def add(role, i):
        parts.append(f"{t('role_' + role)} {parse_unit(cols[i], role) or '—'}")

    if mode in ("CA", "CP"):
        add("time", xi)
    elif p.get("dual_x") and x2kind == "time":
        add("time", x2i)
    if mode in ("CV", "CA"):
        add("cur", yi)
    elif p.get("dual") and y2kind == "cur":
        add("cur", y2i)
    if mode == "CV":
        add("pot", xi)
    elif mode == "CP":
        add("pot", yi)
    elif p.get("dual") and y2kind == "pot":
        add("pot", y2i)
    return "；".join(parts)

p = {"cur_raw": "A", "cur_out": "µA", "norm": False, "area": 1.0, "ref": list(REF)[0], "shift": 0.0, "pot_unit": "auto", "time_raw": "auto"}

AXC = ["black", "follow", "custom"]


def _h(key):
    st.markdown(f"**{t(key)}**")


# ================================================================ 1. 数据
with sb.expander(t("grp_data"), expanded=True):
    _h("sub_cols")
    xi = st.selectbox(t("xcol"), range(len(cols)), index=dx_def, format_func=lambda i: cols[i], key=f"xi_{sig}")
    yi = st.selectbox(t("ycol"), range(len(cols)), index=dy_def, format_func=lambda i: cols[i], key=f"yi_{sig}")
    _want_x, _want_y = WANT_ROLES[mode]
    st.caption(t("detect_note", x=_role_text(xi), y=_role_text(yi)))
    for _lab, _i, _want in ((t("xcol"), xi, _want_x), (t("ycol"), yi, _want_y)):
        if infos[_i]["role"] not in (_want, "other"):
            st.caption("⚠️ " + t("role_mismatch", which=_lab, got=t("role_" + infos[_i]["role"]), want=t("role_" + _want)))
    p["dual"] = st.checkbox(t("dual"), key="dual")
    p["dual_x"] = st.checkbox(t("dual_x"), key="dual_x")

    y2i, y2kind = 0, "other"
    if p["dual"]:
        _y2_def = next((j for j in range(len(cols)) if j not in (xi, yi) and infos[j]["role"] in ("cur", "pot", "charge")), None)
        if _y2_def is None:
            _y2_def = next((j for j in range(len(cols)) if j not in (xi, yi)), min(2, len(cols) - 1))
        y2i = st.selectbox(t("y2col"), range(len(cols)), index=_y2_def, format_func=lambda i: cols[i], key=f"y2i_{sig}")
        guess = {"cur": 0, "pot": 1}.get(infos[y2i]["role"], 2)
        y2kind = st.selectbox(t("y2kind"), ["cur", "pot", "other"], index=guess, format_func=lambda k: t("kind_" + k), key=f"y2kind_{y2i}")
    p["y2kind"] = y2kind

    x2i, x2kind = 0, "other"
    if p["dual_x"]:
        _t_col = pick_col(infos, "time", exclude=(xi,))
        x2_default = _t_col if _t_col is not None else next((j for j in range(len(cols)) if j not in (xi, yi)), min(2, len(cols) - 1))
        x2i = st.selectbox(t("x2col"), range(len(cols)), index=x2_default, format_func=lambda i: cols[i], key=f"x2i_{sig}")
        guess = {"time": 0, "pot": 1}.get(infos[x2i]["role"], 2)
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
        _du = _detected_units_text()
        if _du:
            st.caption(t("unit_auto_note", info=_du))
    if need_time:
        p["time_raw"] = _safe_select(t("time_raw_unit"), ["auto", "s", "ms", "min", "h"], "time_raw", _auto_fmt)
        p["time_unit"] = st.selectbox(t("time_unit"), ["s", "ms", "min", "h"], index=0, key="time_unit")
    else:
        p["time_raw"], p["time_unit"] = "auto", "s"
    if need_cur:
        p["cur_raw"] = _safe_select(t("cur_raw"), ["auto"] + list(CUR), "cur_raw", _auto_fmt)
        p["cur_out"] = st.selectbox(t("cur_out"), list(CUR), index=2, key="cur_out")
        p["norm"] = st.checkbox(t("norm"), key="norm")
        if p["norm"]:
            p["area"] = st.number_input(t("area"), min_value=1e-6, value=0.07, format="%.4f", key="area")
    if need_pot:
        p["pot_unit"] = _safe_select(t("unit_pot_raw"), ["auto", "V", "mV"], "pot_unit_raw", _auto_fmt)
        ref_sel = st.selectbox(
            t("ref"), ["none"] + list(REF) + ["custom"], index=1,
            format_func=lambda r: t("ref_none") if r == "none" else (t("ref_custom") if r == "custom" else r), key="ref",
        )
        p["ref_sel"] = ref_sel
        if ref_sel == "custom":
            p["ref"] = st.text_input(t("ref_custom_name"), value="Ref", key="ref_custom_name").strip() or "Ref"
            p["ref_E"] = st.number_input(t("ref_custom_E"), value=0.0, step=0.001, format="%.3f", key="ref_custom_E")
        elif ref_sel != "none":
            p["ref"] = ref_sel
            p["ref_E"] = REF.get(ref_sel, 0.0)
        else:
            p["ref"] = "none"
            p["ref_E"] = 0.0

        if ref_sel != "none":
            target_opts = ["raw", "she", "rhe"]
            p["pot_target"] = st.selectbox(
                t("pot_target"), target_opts, index=0,
                format_func=lambda k: t("pot_raw") if k == "raw" else (t("to_she") if k == "she" else t("to_rhe")),
                key="pot_target"
            )
        else:
            p["pot_target"] = "raw"

        if p["pot_target"] == "she":
            p["shift"] = p["ref_E"]
        elif p["pot_target"] == "rhe":
            ph = st.number_input(t("ph"), value=7.0, step=0.1, key="ph")
            p["shift"] = p["ref_E"] + 0.0591 * ph
        else:
            p["shift"] = 0.0

        custom_offset_str = st.text_input(t("custom_offset"), value="0.0", key="custom_offset")
        try:
            custom_offset_val = float(custom_offset_str.strip()) if custom_offset_str.strip() else 0.0
        except ValueError:
            custom_offset_val = 0.0
        p["shift"] += custom_offset_val

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

    # 电量与比容量
    st.divider()
    _h("sec_q_capacity")
    p["calc_q"] = st.checkbox(t("calc_q"), value=False, key="calc_q")
    p["calc_cap"] = st.checkbox(t("calc_cap"), value=False, key="calc_cap")
    cpi = 0
    if p["calc_q"] or p["calc_cap"]:
        p["abs_q"] = st.checkbox(t("abs_q"), value=True, key="abs_q")
        p["q_seg"] = _safe_select(t("q_seg"), ["auto", "none"], "q_seg", lambda k: t("q_seg_" + k))
        if mode == "CV":
            p["scan_rate"] = st.number_input(t("scan_rate"), min_value=0.001, value=50.0, step=1.0, key="cv_scan_rate")
            st.caption(t("cv_time_note"))
        if mode == "CP":
            _cp_cur = pick_col(infos, "cur", exclude=(xi, yi))
            p["cp_src"] = _safe_select(
                t("cp_src"), ["col", "const"], f"cp_src_{sig}", lambda k: t("cp_src_" + k), index=0 if _cp_cur is not None else 1,
            )
            if p["cp_src"] == "col":
                _cp_def = _cp_cur if _cp_cur is not None else next((j for j in range(len(cols)) if j not in (xi, yi)), 0)
                cpi = st.selectbox(t("cp_col"), range(len(cols)), index=_cp_def, format_func=lambda i: cols[i], key=f"cpi_{sig}")
                p["cpcur_raw"] = _safe_select(t("cp_cur_raw"), ["auto"] + list(CUR), "cpcur_raw", _auto_fmt)
            else:
                p["cp_const"] = st.number_input(t("cp_const"), value=0.0, step=0.1, format="%.4f", key="cp_const")
        p["mass_unit"] = _safe_select(t("mass_unit"), ["mg", "g", "µg"], "mass_unit")
        p["active_mass"] = st.number_input(t("active_mass"), min_value=0.0, value=0.0, step=0.1, format="%.6g", key="cv_active_mass")
        p["mass_g"] = p["active_mass"] * {"g": 1.0, "mg": 1e-3, "µg": 1e-6}.get(p["mass_unit"], 1e-3)
        st.caption(t("calc_note"))

# ================================================================ 3. 坐标轴
with sb.expander(t("grp_axes")):
    p["title"] = st.text_input(t("fig_title"), key="fig_title")
    _h("ax_x")
    p["lab_x"] = st.text_input(t("lab_x"), key="lab_x", help=t("lab_help"))
    r1, r2 = st.columns(2)
    xmin = r1.text_input(t("xmin"), key="xmin")
    xmax = r2.text_input(t("xmax"), key="xmax")
    stx = st.text_input(t("stepx"), key="step_x")
    axc_x = st.selectbox(t("ax_color"), AXC, index=0, format_func=lambda k: t("yc_" + k), key="axc_x")
    xb_cust = st.color_picker(t("color_xb"), "#000000", key="xb_col") if axc_x == "custom" else "#000000"

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

    st.divider()
    _h("ax_y")
    p["lab_y"] = st.text_input(t("lab_y"), key="lab_y", help=t("lab_help"))
    r1, r2 = st.columns(2)
    ymin = r1.text_input(t("ymin"), key="ymin")
    ymax = r2.text_input(t("ymax"), key="ymax")
    sty = st.text_input(t("stepy"), key="step_y")
    axc_y = st.selectbox(t("ax_color"), AXC, index=(1 if p["dual"] else 0), format_func=lambda k: t("yc_" + k), key=f"axc_y_{p['dual']}")
    yl_cust = st.color_picker(t("color_yl"), "#000000", key="yl_col") if axc_y == "custom" else "#000000"

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

# ================================================================ 5. 自定义文本框
with sb.expander(t("grp_textbox")):
    num_tb = int(st.number_input(t("num_tb"), min_value=0, max_value=10, value=0, step=1, key="num_tb"))
    textboxes = []
    for i in range(num_tb):
        st.markdown(f"**{t('tb_title', n=i+1)}**")
        tb_txt = st.text_input(t("tb_text"), key=f"tb_txt_{i}")
        c_coord, c_x, c_y = st.columns([1.5, 1, 1])
        tb_coord = c_coord.selectbox(t("tb_coord"), ["data", "rel"], format_func=lambda k: t("coord_" + k), key=f"tb_coord_{i}")
        tb_x = c_x.number_input(t("tb_x"), value=0.5 if tb_coord == "rel" else 0.0, step=0.05 if tb_coord == "rel" else 0.01, format="%.3f" if tb_coord == "rel" else "%.4g", key=f"tb_x_{i}")
        tb_y = c_y.number_input(t("tb_y"), value=0.5 if tb_coord == "rel" else 0.0, step=0.05 if tb_coord == "rel" else 0.01, format="%.3f" if tb_coord == "rel" else "%.4g", key=f"tb_y_{i}")
        
        c_fs, c_rot, c_clr = st.columns([1, 1, 1])
        tb_fs = c_fs.number_input(t("tb_fs"), min_value=4, max_value=36, value=9, step=1, key=f"tb_fs_{i}")
        tb_rot = c_rot.number_input(t("tb_rot"), min_value=-360, max_value=360, value=0, step=15, key=f"tb_rot_{i}")
        tb_color = c_clr.color_picker(t("tb_color"), "#000000", key=f"tb_clr_{i}")
        tb_frame = st.selectbox(t("tb_frame"), ["none", "box", "solid"], format_func=lambda k: t("frame_" + k), key=f"tb_frame_{i}")
        tb_on = st.multiselect(t("tb_on"), ["main", "q", "cap"], default=["main", "q", "cap"], format_func=lambda k: t("fig_" + k), key=f"tb_on_{i}")
        
        if tb_txt.strip():
            textboxes.append({
                "text": tb_txt.strip(),
                "coord": tb_coord,
                "x": float(tb_x),
                "y": float(tb_y),
                "fs": int(tb_fs),
                "rot": int(tb_rot),
                "color": tb_color,
                "frame": tb_frame,
                "on": [v for v in tb_on if v in ("main", "q", "cap")],
            })
    p["textboxes"] = textboxes

def _axis_color(mode_, custom, follow):
    return "#000000" if mode_ == "black" else (follow if mode_ == "follow" else custom)


_c0l, _c0r = p["colors"][0]
p["xb_col"] = _axis_color(axc_x, xb_cust, _c0l)
p["yl_col"] = _axis_color(axc_y, yl_cust, _c0l)
p["yr_col"] = _axis_color(axc_y2, yr_cust, _c0r or "#000000") if p["dual"] else "#000000"
p["xt_col"] = _axis_color(axc_x2, xt_cust, p["colors_x2"][0] or "#000000") if p["dual_x"] else "#000000"

# ================================================================ 6. 图形外观
with sb.expander(t("grp_fig")):
    p["w"] = st.selectbox(t("width"), [85, 120, 175], format_func=lambda w: t(f"w{w}"), key="width")
    p["h"] = st.slider(t("height"), 40, 120, 65, key="height")
    p["fs"] = st.slider(t("fontsize"), 6, 14, 9, key="fs")

# ================================================================ 参考线（主图 / 电量图 / 比容量图各自设置）
oq, ocap = None, None


def _ref_lines(px):
    n = int(st.number_input(t("rl_n"), min_value=0, max_value=10, value=0, step=1, key=f"{px}_nref"))
    out = []
    for i in range(n):
        st.markdown(f"**{t('rl_title', n=i + 1)}**")
        c1, c2 = st.columns(2)
        orient = _safe_select(t("rl_orient"), ["h", "v"], f"{px}_ro_{i}", lambda k: t("rl_orient_" + k), where=c1)
        pos_txt = c2.text_input(t("rl_pos"), key=f"{px}_rp_{i}")
        c3, c4 = st.columns(2)
        color = c3.color_picker(t("rl_color"), "#808080", key=f"{px}_rc_{i}")
        style = _safe_select(t("rl_style"), ["dashed", "dotted", "dashdot", "solid"], f"{px}_rs_{i}", lambda k: t("rl_style_" + k), where=c4)
        axis = "main"
        if px == "main" and orient == "h" and p["dual"]:
            axis = _safe_select(t("rl_axis"), ["y", "y2"], f"{px}_ra_h_{i}", lambda k: t("ax_y") if k == "y" else t("ax_y2"))
        if px == "main" and orient == "v" and p["dual_x"]:
            axis = _safe_select(t("rl_axis"), ["x", "x2"], f"{px}_ra_v_{i}", lambda k: t("ax_x") if k == "x" else t("ax_x2"))
        try:
            v = parse_float(pos_txt)
        except ValueError:
            st.caption("⚠️ " + t("warn_num"))
            continue
        if v is not None:
            out.append({"orient": orient, "pos": v, "color": color, "style": style, "axis": axis})
    return out


with sb.expander(t("grp_reflines")):
    _h("fig_main")
    p["reflines"] = _ref_lines("main")
    if p.get("calc_q"):
        st.divider()
        _h("fig_q")
        _q_lines = _ref_lines("q")
    if p.get("calc_cap"):
        st.divider()
        _h("fig_cap")
        _cap_lines = _ref_lines("cap")


def _fig_opts(px, lines):
    # 电量图 / 比容量图的独立设置：和主图一样的选项（标题、轴标签、范围、间隔、轴颜色、曲线颜色、图例、刻度、尺寸）
    o = {"reflines": lines}
    o["title"] = st.text_input(t("fig_title"), key=f"{px}_title")
    _h("ax_x")
    o["lab_x"] = st.text_input(t("lab_x"), key=f"{px}_lab_x", help=t("lab_help"))
    r1, r2 = st.columns(2)
    xmin_ = r1.text_input(t("xmin"), key=f"{px}_xmin")
    xmax_ = r2.text_input(t("xmax"), key=f"{px}_xmax")
    stx_ = st.text_input(t("stepx"), key=f"{px}_stepx")
    axc_x_ = _safe_select(t("ax_color"), AXC, f"{px}_axc_x", lambda k: t("yc_" + k))
    xcust = st.color_picker(t("color_xb"), "#000000", key=f"{px}_xcol") if axc_x_ == "custom" else "#000000"
    st.divider()
    _h("ax_y")
    o["lab_y"] = st.text_input(t("lab_y"), key=f"{px}_lab_y", help=t("lab_help"))
    r1, r2 = st.columns(2)
    ymin_ = r1.text_input(t("ymin"), key=f"{px}_ymin")
    ymax_ = r2.text_input(t("ymax"), key=f"{px}_ymax")
    sty_ = st.text_input(t("stepy"), key=f"{px}_stepy")
    axc_y_ = _safe_select(t("ax_color"), AXC, f"{px}_axc_y", lambda k: t("yc_" + k))
    ycust = st.color_picker(t("color_yl"), "#000000", key=f"{px}_ycol") if axc_y_ == "custom" else "#000000"
    st.divider()
    _h("sub_curves")
    follow = st.checkbox(t("follow_main"), value=True, key=f"{px}_follow")
    cl_list = []
    for k, name in enumerate(dfs):
        if follow:
            cl_list.append(p["colors"][k][0])
        else:
            cl_list.append(st.color_picker(t("color_curve", name=Path(name).stem), COLORS[k % 7], key=f"{px}_c_{name}"))
    o["colors"] = [(c, None) for c in cl_list]
    o["scatter"] = st.checkbox(t("scatter"), value=False, key=f"{px}_scatter")
    st.divider()
    _h("sub_legend")
    o["legend"] = st.checkbox(t("legend"), value=len(dfs) > 1, key=f"{px}_legend")
    o["legend_loc"] = _safe_select(
        t("legend_loc"), ["best", "upper right", "upper left", "lower right", "lower left", "manual"], f"{px}_leg_loc",
        lambda k: t("leg_manual") if k == "manual" else k,
    )
    o["legend_xy"] = (0.02, 0.98)
    if o["legend_loc"] == "manual":
        lc = st.columns(2)
        o["legend_xy"] = (
            lc[0].number_input(t("leg_x"), value=0.02, step=0.05, format="%.2f", key=f"{px}_leg_x"),
            lc[1].number_input(t("leg_y"), value=0.98, step=0.05, format="%.2f", key=f"{px}_leg_y"),
        )
        st.caption(t("leg_manual_note"))
    o["legend_ncol"] = int(st.number_input(t("leg_ncol"), min_value=1, max_value=6, value=1, step=1, key=f"{px}_leg_ncol"))
    o["legend_frame"] = st.checkbox(t("leg_frame"), value=False, key=f"{px}_leg_frame")
    st.caption(t("sec_leg_text"))
    o["custom_legends"] = [(st.text_input(t("leg_text_single", name=Path(n).stem), key=f"{px}_leg_{n}").strip(), "") for n in dfs]
    st.divider()
    _h("sub_ticks_common")
    st.caption(t("tick_marks"))
    cc = st.columns(4)
    o["tick_marks"] = {s: cc[i].checkbox(t("side_" + s), value=True, key=f"{px}_tm_{s}") for i, s in enumerate(SIDES)}
    st.caption(t("tick_nums"))
    cc = st.columns(4)
    nd = {"bottom": True, "left": True, "top": False, "right": False}
    o["tick_nums"] = {s: cc[i].checkbox(t("side_" + s), value=nd[s], key=f"{px}_tn_{s}") for i, s in enumerate(SIDES)}
    o["minor"] = st.checkbox(t("minor"), value=True, key=f"{px}_minor")
    o["tick_dir"] = _safe_select(t("tdir"), ["in", "out"], f"{px}_tdir", lambda d: t("dir_" + d))
    st.divider()
    o["w"] = _safe_select(t("width"), [85, 120, 175], f"{px}_width", lambda w: t(f"w{w}"))
    o["h"] = st.slider(t("height"), 40, 120, 65, key=f"{px}_height")
    o["fs"] = st.slider(t("fontsize"), 6, 14, 9, key=f"{px}_fs")
    o["_raw"] = dict(xmin=xmin_, xmax=xmax_, ymin=ymin_, ymax=ymax_, stepx=stx_, stepy=sty_)
    o["_axc"] = (axc_x_, xcust, axc_y_, ycust)
    return o


if p.get("calc_q"):
    with sb.expander(t("grp_qfig")):
        oq = _fig_opts("q", _q_lines)
if p.get("calc_cap"):
    with sb.expander(t("grp_capfig")):
        ocap = _fig_opts("cap", _cap_lines)


def _finish_opts(o):
    # 解析文本框里的范围和间隔，并确定两个轴的颜色（“跟随曲线”取第一个文件的曲线颜色）
    raw = o.pop("_raw")
    o["xlim"] = (parse_float(raw["xmin"]), parse_float(raw["xmax"]))
    o["ylim"] = (parse_float(raw["ymin"]), parse_float(raw["ymax"]))
    o["xstep"], o["ystep"] = parse_float(raw["stepx"]), parse_float(raw["stepy"])
    if any(v is not None and v <= 0 for v in (o["xstep"], o["ystep"])):
        raise ValueError("step")
    ax_x_mode, x_cust, ax_y_mode, y_cust = o.pop("_axc")
    c0 = o["colors"][0][0]
    o["xb_col"] = _axis_color(ax_x_mode, x_cust, c0)
    o["yl_col"] = _axis_color(ax_y_mode, y_cust, c0)
    return o


# ================================================================ 7. AI 分析设置
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
    for _o in (oq, ocap):
        if _o is not None:
            _finish_opts(_o)
except ValueError:
    st.warning(t("warn_num"))
    st.stop()

results, rows = [], []
for name, df in dfs.items():
    d = df.iloc[int(p_start) : (int(p_end) or None)]
    idx = d.index.to_numpy().copy()
    # 每个文件单独找列：先按列名，找不到再按角色，最后才按位置；单位也逐文件识别
    infos_f = [col_info(c) for c in df.columns]
    xi_f = resolve_col(df, infos_f, cols[xi], infos[xi]["role"], xi)
    yi_f = resolve_col(df, infos_f, cols[yi], infos[yi]["role"], yi)
    y2i_f = resolve_col(df, infos_f, cols[y2i], infos[y2i]["role"], y2i) if p["dual"] else y2i
    x2i_f = resolve_col(df, infos_f, cols[x2i], infos[x2i]["role"], x2i) if p["dual_x"] else x2i
    cn = lambda j: str(df.columns[j])
    pf = dict(p)
    pf["cur_raw"] = eff_unit(p["cur_raw"], cn(yi_f) if mode in ("CV", "CA") else "", "cur", "A")
    _pot_col = cn(xi_f) if mode == "CV" else (cn(yi_f) if mode == "CP" else "")
    pf["pot_scale"] = 1e-3 if eff_unit(p["pot_unit"], _pot_col, "pot", "V") == "mV" else 1.0
    pf["t_scale"] = TSEC[eff_unit(p["time_raw"], cn(xi_f) if mode != "CV" else "", "time", "s")]
    pf2 = dict(pf)
    if p["dual"]:
        pf2["cur_raw"] = eff_unit(p["cur_raw"], cn(y2i_f), "cur", "A")
        pf2["pot_scale"] = 1e-3 if eff_unit(p["pot_unit"], cn(y2i_f), "pot", "V") == "mV" else 1.0
    pfx = dict(pf)
    if p["dual_x"]:
        pfx["t_scale"] = TSEC[eff_unit(p["time_raw"], cn(x2i_f), "time", "s")]
        pfx["pot_scale"] = 1e-3 if eff_unit(p["pot_unit"], cn(x2i_f), "pot", "V") == "mV" else 1.0

    x, y_raw = transform(d.iloc[:, xi_f].to_numpy(float).copy(), d.iloc[:, yi_f].to_numpy(float).copy(), mode, pf)
    if mode != "CV":
        x = time_conv(x * pf["t_scale"], p["time_unit"])
    y2_raw = transform_y2(d.iloc[:, y2i_f].to_numpy(float).copy(), y2kind, pf2) if p["dual"] else None
    x2_raw = transform_x2(d.iloc[:, x2i_f].to_numpy(float).copy(), x2kind, pfx) if p["dual_x"] else None

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
        if mode != "CV":
            x = fix(x, keep)
        if x2 is not None and x2kind == "time":
            x2 = fix(x2, keep)

    y, y2 = y_raw.copy(), (y2_raw.copy() if y2_raw is not None else None)
    if p["sm_method"] != "none":
        y[keep] = smooth_1d(y_raw[keep], p["sm_method"], sm_win, sm_poly, sm_sigma)
        if y2 is not None and p["sm_right"]:
            y2[keep] = smooth_1d(y2_raw[keep], p["sm_method"], sm_win, sm_poly, sm_sigma)

    res_item = {
        "name": Path(name).stem,
        "idx": idx,
        "x": x,
        "y": y,
        "y_raw": y_raw,
        "y2": y2,
        "y2_raw": y2_raw,
        "keep": keep,
        "fit": None,
        "fit_info": None,
        "ylab": short(cols[yi]),
        "y2lab": short(cols[y2i]),
        "y2name": cols[y2i],
        "x2": x2,
        "x2name": cols[x2i],
    }

    # ------------------------------------------------ 电量与比容量
    if p.get("calc_q") or p.get("calc_cap"):
        try:
            n_all = len(d)
            e_v, t_sec = None, None
            if mode == "CV":
                e_v = d.iloc[:, xi_f].to_numpy(float) * pf["pot_scale"]
                i_amp = d.iloc[:, yi_f].to_numpy(float) * CUR[pf["cur_raw"]]
                _tc = pick_col(infos_f, "time")
                if _tc is not None:
                    t_sec = d.iloc[:, _tc].to_numpy(float) * TSEC[parse_unit(cn(_tc), "time") or "s"]
            else:
                t_sec = d.iloc[:, xi_f].to_numpy(float) * pf["t_scale"]
                if mode == "CA":
                    i_amp = d.iloc[:, yi_f].to_numpy(float) * CUR[pf["cur_raw"]]
                elif p.get("cp_src") == "const":
                    i_amp = np.full(n_all, p.get("cp_const", 0.0) * 1e-3)
                else:
                    cpi_f = resolve_col(df, infos_f, cols[cpi], infos[cpi]["role"], cpi)
                    i_amp = d.iloc[:, cpi_f].to_numpy(float) * CUR[eff_unit(p.get("cpcur_raw", "auto"), cn(cpi_f), "cur", "A")]
            res_item.update(
                compute_charge(
                    mode, t_sec, i_amp, e_v, keep, p.get("abs_q", True), p.get("q_seg", "auto") == "auto",
                    p.get("mass_g", 0.0), p.get("scan_rate", 50.0) / 1000.0,
                )
            )
        except ValueError as e:
            st.warning(t("calc_fail", name=Path(name).stem, err=t("calc_err_" + str(e)) if ("calc_err_" + str(e)) in TR["zh"] else e))

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

    res_item["fit"] = fit
    res_item["fit_info"] = fit_info
    rows.append(row)
    results.append(res_item)

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
    if r.get("q_segs"):
        a["charge_segments"] = [{k2: (_sig(v2) if isinstance(v2, float) else v2) for k2, v2 in s2.items()} for s2 in r["q_segs"][:20]]
    analysis.append(a)
for r, a in zip(results, analysis):
    r["peaks"] = a.get("cv_peaks")

# ---------------------------------------------------------------- 视图渲染
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

# ---------------------------------------------------------------- 电量图与比容量图独立渲染区域
fig_q_obj, fig_cap_obj = None, None

if p.get("calc_q"):
    st.divider()
    st.subheader(t("q_fig_title"))
    fig_q_obj = build_q_fig(results, mode, {**p, **oq})
    st.image(export_bytes(fig_q_obj, "png", 250), width=640)

if p.get("calc_cap"):
    st.divider()
    st.subheader(t("cap_fig_title"))
    fig_cap_obj = build_cap_fig(results, mode, {**p, **ocap})
    st.image(export_bytes(fig_cap_obj, "png", 250), width=640)

if p.get("calc_q") or p.get("calc_cap"):
    st.divider()
    st.subheader(t("q_summary"))
    _qrows = []
    for r in results:
        for s in r.get("q_segs", []):
            _qrows.append(
                {
                    t("qs_file"): r["name"], t("qs_seg"): s["seg"], t("qs_dir"): {1: "+", -1: "−"}.get(s["dir"], ""),
                    t("qs_n"): s["n"], t("qs_dur"): round(s["dur_s"], 3), t("qs_q"): float(f"{s['q']:.5g}"),
                    f"{t('qs_cap')} / ({r['cap_unit'].replace('$^{-1}$', '⁻¹')})": float(f"{s['cap']:.5g}"),
                }
            )
    if _qrows:
        st.dataframe(pd.DataFrame(_qrows), hide_index=True)
    else:
        st.caption(t("qs_none"))

# ---------------------------------------------------------------- 导出与下载选项（全图表支持）
st.divider()
st.subheader(t("sec_export"))
e1, e2, e3 = st.columns([1, 1, 2])
fmt = e1.selectbox(t("fmt"), ["png", "pdf", "svg", "tiff"], format_func=str.upper, key="fmt")
dpi = e2.selectbox(t("dpi"), [300, 600, 1200], index=1, key="dpi")
fname = e3.text_input(t("fname"), value="figure", key="fname")

b1, b2, b3 = st.columns(3)

# 导出主图
final_main = build_fig(results, mode, p, show_removed=False)
payload_main = export_bytes(final_main, fmt, dpi)
plt.close(final_main)
b1.download_button(t("dl"), payload_main, file_name=f"{fname.strip() or 'figure'}_main.{fmt}", mime=MIME[fmt], type="primary", key="dl_main")

# 导出电量图
if fig_q_obj is not None:
    payload_q = export_bytes(fig_q_obj, fmt, dpi)
    plt.close(fig_q_obj)
    b2.download_button(t("dl_q"), payload_q, file_name=f"{fname.strip() or 'figure'}_charge.{fmt}", mime=MIME[fmt], type="secondary", key="dl_q")

# 导出比容量图
if fig_cap_obj is not None:
    payload_cap = export_bytes(fig_cap_obj, fmt, dpi)
    plt.close(fig_cap_obj)
    b3.download_button(t("dl_cap"), payload_cap, file_name=f"{fname.strip() or 'figure'}_capacity.{fmt}", mime=MIME[fmt], type="secondary", key="dl_cap")

st.caption(t("dl_hint"))

# ---------------------------------------------------------------- AI 分析模块
st.divider()
st.subheader(t("sec_ai"))
st.caption(t("ai_intro"))
st.caption(t("ai_warn"))

xl_auto, yl_auto = axis_labels(mode, p)
payload = {
    "technique": {"CV": "cyclic voltammetry", "CA": "chronoamperometry", "CP": "chronopotentiometry"}[mode],
    "axes": {"x": p["lab_x"] or xl_auto, "y_left": p["lab_y"] or yl_auto},
    "processing": {
        "reference_electrode": None if p["ref"] == "none" else p["ref"],
        "potential_target": p.get("pot_target"),
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
                    text = call_gemini(api_key, ai_model.strip(), system_text, user_text)
                st.session_state["ai_result"] = (sig_now, text)
            except Exception as e:
                st.error(t("ai_fail", err=f"{type(e).__name__}: {str(e)[:300]}"))

saved = st.session_state.get("ai_result")
if saved:
    if saved[0] != sig_now:
        st.caption(t("ai_stale"))
    st.markdown(saved[1])