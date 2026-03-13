#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np

# чтобы можно было импортировать lib.* при запуске как скрипта
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from lib.postprocess_md import (
    parse_xtb_trj,
    get_box_from_coord,
    compute_rdf_pair,
)


def load_mapping(root: Path) -> dict | None:
    """
    Ищет system.mapping.json внутри архива и возвращает его содержимое.
    """
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith(".mapping.json"):
                mp = Path(dirpath) / fn
                try:
                    return json.loads(mp.read_text(encoding="utf-8"))
                except Exception:
                    return None
    return None


def compute_msd_per_type(
    coords_list: List[np.ndarray],
    frames_meta: List[dict[str, Any]],
    mapping: dict,
) -> Tuple[dict, dict]:
    """
    Считает MSD по типам молекул.

    Возвращает:
      msd_series: {type_name: [{"time": t_ps, "msd": value}, ...]}
      msd_windows: {type_name: [{"center_time": t_ps, "msd": avg_msd}, ...]}
    """
    molecules = mapping.get("molecules", [])
    n_frames = len(coords_list)
    if n_frames == 0 or not molecules:
        return {}, {}

    # Для каждого типа: список инстансов (start,end)
    type_instances: dict[str, List[Tuple[int, int]]] = {}
    for m in molecules:
        name = str(m.get("name", "X"))
        insts = m.get("instances", [])
        for inst in insts:
            s = int(inst["start"])
            e = int(inst["end"])
            type_instances.setdefault(name, []).append((s, e))

    if not type_instances:
        return {}, {}

    # Центры масс молекул по кадрам (без реальных масс, просто геометрический центр)
    msd_series: dict[str, List[dict[str, float]]] = {}
    msd_windows: dict[str, List[dict[str, float]]] = {}

    for name, inst_list in type_instances.items():
        n_inst = len(inst_list)
        if n_inst == 0:
            continue

        # positions[inst_idx, frame, coord]
        positions = np.zeros((n_inst, n_frames, 3), dtype=float)
        for fi, coords in enumerate(coords_list):
            coords = np.asarray(coords, dtype=float)
            for ii, (s, e) in enumerate(inst_list):
                frag = coords[s : e + 1]
                positions[ii, fi] = frag.mean(axis=0)

        # MSD(t) = <|R_i(t) - R_i(0)|^2>_i
        pos0 = positions[:, 0, :]  # (n_inst,3)
        disp = positions - pos0[:, None, :]  # (n_inst, n_frames, 3)
        sq = np.sum(disp * disp, axis=2)  # (n_inst, n_frames)
        msd = sq.mean(axis=0)  # (n_frames,)

        series = []
        for fi in range(n_frames):
            t = float(frames_meta[fi]["time_ps"])
            series.append({"time": t, "msd": float(msd[fi])})
        msd_series[name] = series

        # Скользящее окно по MSD: 10% длины, шаг 2%, усредняем внутри окна
        win_len = max(1, int(0.10 * n_frames))
        step = max(1, int(0.02 * n_frames))
        if win_len > n_frames:
            win_len = n_frames

        windows: List[dict[str, float]] = []
        for start in range(0, n_frames - win_len + 1, step):
            end = start + win_len
            window_vals = msd[start:end]
            avg_msd = float(window_vals.mean())
            center_idx = start + win_len // 2
            center_time = float(frames_meta[center_idx]["time_ps"])
            windows.append({"center_time": center_time, "msd": avg_msd})

        msd_windows[name] = windows

    return msd_series, msd_windows



def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def find_file(root: Path, name: str) -> Path | None:
    for dirpath, _, filenames in os.walk(root):
        if name in filenames:
            return Path(dirpath) / name
    return None


def generate_index_html(
    energy: list[dict[str, Any]] | None,
    hbonds: list[dict[str, Any]] | None,
    rdf: list[dict[str, Any]] | None,
    sys_params: list[dict[str, Any]] | None,
) -> str:
    # Подготовка данных для встраивания в JS
    def to_number(x: str) -> float:
        try:
            return float(x)
        except Exception:
            return float("nan")

    energy_js = []
    if energy:
        for row in energy:
            energy_js.append(
                {
                    "time": to_number(row.get("time_ps", row.get("time", "0"))),
                    "energy": to_number(
                        row.get("energy", row.get("energy_au", row.get("energy_kcalmol", "0")))
                    ),
                }
            )

    hbonds_js = []
    if hbonds:
        for row in hbonds:
            hbonds_js.append(
                {
                    "time": to_number(row.get("time_ps", row.get("time", "0"))),
                    "count": to_number(row.get("n_hbonds", row.get("count", "0"))),
                    "avg_distance": to_number(row.get("avg_distance", "nan")),
                    "avg_angle": to_number(row.get("avg_angle", "nan")),
                }
            )

    rdf_js = []
    if rdf:
        for row in rdf:
            rdf_js.append(
                {
                    "r": to_number(row.get("r", "0")),
                    "g_OH": to_number(row.get("g_OH", "nan")),
                    "g_OO": to_number(row.get("g_OO", "nan")),
                    "g_NH": to_number(row.get("g_NH", "nan")),
                    "g_NO": to_number(row.get("g_NO", "nan")),
                    "g_NN": to_number(row.get("g_NN", "nan")),
                }
            )

    sys_js = []
    sys_keys: list[str] = []
    if sys_params:
        # первая строка определяет набор параметров (кроме времени)
        first = sys_params[0]
        time_key = "time_ps" if "time_ps" in first else "time"
        sys_keys = [k for k in first.keys() if k not in {time_key}]
        for row in sys_params:
            item: dict[str, Any] = {
                "time": to_number(row.get(time_key, "0")),
            }
            for k in sys_keys:
                item[k] = to_number(row.get(k, "nan"))
            sys_js.append(item)

    energy_json = json.dumps(energy_js)
    hbonds_json = json.dumps(hbonds_js)
    rdf_json = json.dumps(rdf_js)
    sys_json = json.dumps(sys_js)
    sys_keys_json = json.dumps(sys_keys)

    # HTML с AnyChart – главная страница (обзор)
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>MD Analysis Report</title>
  <script src="https://cdn.anychart.com/releases/8.14.1/js/anychart-base.min.js"></script>
  <style>
    body {{
      font-family: sans-serif;
      margin: 0;
      padding: 0;
    }}
    .chart-row {{
      display: flex;
      flex-wrap: wrap;
    }}
    .chart-container {{
      box-sizing: border-box;
      padding: 10px;
      width: 50%;
      min-width: 400px;
      height: 400px;
    }}
    h1 {{
      padding: 10px;
    }}
  </style>
</head>
<body>
  <h1>MD Analysis – Overview</h1>
  <p style="padding: 0 10px 10px 10px;">
    RDF по отдельным парам: 
    <a href="rdf_OH.html">OH</a>,
    <a href="rdf_OO.html">OO</a>,
    <a href="rdf_NH.html">NH</a>,
    <a href="rdf_NO.html">NO</a>,
    <a href="rdf_NN.html">NN</a>.
    Анализ молекул: <a href="msd.html">MSD по типам</a>.
  </p>
  <div class="chart-row">
    <div id="energy_chart" class="chart-container"></div>
    <div id="hbonds_chart" class="chart-container"></div>
  </div>
  <div class="chart-row">
    <div id="rdf_chart" class="chart-container"></div>
    <div id="sys_chart" class="chart-container"></div>
  </div>

  <script>
    var energyData = {energy_json};
    var hbondsData = {hbonds_json};
    var rdfData = {rdf_json};
    var sysData = {sys_json};
    var sysKeys = {sys_keys_json};

    anychart.onDocumentReady(function() {{
      // Энергия
      if (energyData.length > 0) {{
        var eChart = anychart.line();
        eChart.title("Energy vs Time");
        eChart.xAxis().title("time, ps");
        eChart.yAxis().title("energy");
        eChart.data(energyData.map(function(p) {{
          return {{ x: p.time, value: p.energy }};
        }}));
        eChart.container("energy_chart");
        eChart.draw();
      }} else {{
        document.getElementById("energy_chart").innerText = "No energy data.";
      }}

      // Водородные связи
      if (hbondsData.length > 0) {{
        var hChart = anychart.line();
        hChart.title("Hydrogen bonds vs Time");
        hChart.xAxis().title("time, ps");
        hChart.yAxis().title("count / avg");

        var sCount = hChart.line(hbondsData.map(function(p) {{
          return {{ x: p.time, value: p.count }};
        }}));
        sCount.name("count");

        var sDist = hChart.line(hbondsData.map(function(p) {{
          return {{ x: p.time, value: p.avg_distance }};
        }}));
        sDist.name("avg distance");

        var sAngle = hChart.line(hbondsData.map(function(p) {{
          return {{ x: p.time, value: p.avg_angle }};
        }}));
        sAngle.name("avg angle");

        hChart.legend(true);
        hChart.container("hbonds_chart");
        hChart.draw();
      }} else {{
        document.getElementById("hbonds_chart").innerText = "No H-bond data.";
      }}

      // RDF
      if (rdfData.length > 0) {{
        var rChart = anychart.line();
        rChart.title("Radial distribution functions");
        rChart.xAxis().title("r, Å");
        rChart.yAxis().title("g(r)");

        function addSeries(key, name) {{
          var s = rChart.line(rdfData.map(function(p) {{
            return {{ x: p.r, value: p[key] }};
          }}));
          s.name(name);
        }}
        addSeries("g_OH", "g_OH");
        addSeries("g_OO", "g_OO");
        addSeries("g_NH", "g_NH");
        addSeries("g_NO", "g_NO");
        addSeries("g_NN", "g_NN");

        rChart.legend(true);
        rChart.container("rdf_chart");
        rChart.draw();
      }} else {{
        document.getElementById("rdf_chart").innerText = "No RDF data.";
      }}

      // Параметры системы
      if (sysData.length > 0 && sysKeys.length > 0) {{
        var sChart = anychart.line();
        sChart.title("System parameters vs Time");
        sChart.xAxis().title("time, ps");
        sChart.yAxis().title("value");

        sysKeys.forEach(function(key) {{
          var s = sChart.line(sysData.map(function(p) {{
            return {{ x: p.time, value: p[key] }};
          }}));
          s.name(key);
        }});

        sChart.legend(true);
        sChart.container("sys_chart");
        sChart.draw();
      }} else {{
        document.getElementById("sys_chart").innerText = "No system-parameters data.";
      }}
    }});
  </script>
</body>
</html>
"""
    return html


def generate_rdf_pair_html(
    pair_key: str,
    pair_label: str,
    rdf_rows: list[dict[str, Any]] | None,
    window_series: list[dict[str, Any]] | None,
) -> str:
    """
    Отдельная страница для одной пары (например, OH).
    Используем только суммарный RDF (по всей траектории) и отмечаем пики.
    """
    def to_number(x: str) -> float:
        try:
            return float(x)
        except Exception:
            return float("nan")

    data = []
    if rdf_rows:
        for row in rdf_rows:
            r = to_number(row.get("r", "0"))
            g = to_number(row.get(pair_key, "nan"))
            data.append({"r": r, "g": g})

    # поиск пиков на глобальном RDF
    peaks = []
    for i in range(1, len(data) - 1):
        g_prev = data[i - 1]["g"]
        g_curr = data[i]["g"]
        g_next = data[i + 1]["g"]
        if g_curr > g_prev and g_curr > g_next and g_curr > 0.5:
            peaks.append({"r": data[i]["r"], "g": g_curr})

    data_json = json.dumps(data)
    peaks_json = json.dumps(peaks)
    windows_json = json.dumps(window_series or [])

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>RDF {pair_label}</title>
  <script src="https://cdn.anychart.com/releases/8.14.1/js/anychart-base.min.js"></script>
  <script src="https://cdn.anychart.com/releases/8.14.1/js/anychart-heatmap.min.js"></script>
  <script src="https://cdn.anychart.com/releases/8.14.1/js/anychart-cartesian.min.js"></script>
  <style>
    body {{
      font-family: sans-serif;
      margin: 0;
      padding: 0;
    }}
    #chart_container {{
      width: 100%;
      height: 500px;
    }}
    #window_chart {{
      width: 100%;
      height: 500px;
    }}
    #heatmap_chart {{
      width: 100%;
      height: 500px;
    }}
    #ci_chart {{
      width: 100%;
      height: 500px;
    }}
    #window_controls {{
      padding: 10px;
    }}
    a.back {{
      display: inline-block;
      margin: 10px;
    }}
  </style>
</head>
<body>
  <a href="index.html" class="back">&larr; Back to overview</a>
  <h1 style="padding: 0 10px 10px 10px;">RDF {pair_label}</h1>
  <div id="chart_container"></div>

  <h2 style="padding: 10px 10px 0 10px;">Heatmap RDF {pair_label} (r vs time)</h2>
  <div id="heatmap_chart"></div>

  <div id="window_controls">
    <label for="window_select">Window (10% length, step 2%): </label>
    <select id="window_select"></select>
  </div>
  <div id="window_chart"></div>

  <h2 style="padding: 10px 10px 0 10px;">RDF {pair_label} with confidence intervals</h2>
  <div id="ci_chart"></div>

  <script>
    var rdfData = {data_json};
    var peaks = {peaks_json};
    var windowSeries = {windows_json};

    anychart.onDocumentReady(function() {{
      // Глобальный RDF
      if (rdfData.length === 0) {{
        document.getElementById("chart_container").innerText = "No RDF data for {pair_label}.";
      }} else {{
        var chart = anychart.line();
        chart.title("RDF {pair_label} (full trajectory)");
        chart.xAxis().title("r, Å");
        chart.yAxis().title("g(r)");

        var sRdf = chart.line(rdfData.map(function(p) {{
          return {{ x: p.r, value: p.g }};
        }}));
        sRdf.name("g_{pair_label}");

        // пиковые точки
        if (peaks.length > 0) {{
          var sPeaks = chart.marker(peaks.map(function(p) {{
            return {{ x: p.r, value: p.g }};
          }}));
          sPeaks.name("peaks");
          sPeaks.normal().shape("circle").size(5).fill("red").stroke(null);
        }}

        chart.legend(true);
        chart.container("chart_container");
        chart.draw();
      }}

      // Тепловая карта: r vs time, g(r) как цвет
      (function() {{
        if (!windowSeries || windowSeries.length === 0) {{
          document.getElementById("heatmap_chart").innerText = "No windowed RDF data for {pair_label}.";
          return;
        }}
        var heatData = [];
        windowSeries.forEach(function(w) {{
          if (!w || !w.series) return;
          var t = (w.center_time !== undefined && !isNaN(w.center_time)) ? w.center_time : null;
          if (t === null) return;
          w.series.forEach(function(p) {{
            if (p.r === undefined || p.g === undefined) return;
            heatData.push({{ x: p.r, y: t, heat: p.g }});
          }});
        }});
        if (heatData.length === 0) {{
          document.getElementById("heatmap_chart").innerText = "No windowed RDF data for {pair_label}.";
          return;
        }}
        var hChart = anychart.heatMap(heatData);
        hChart.title("RDF {pair_label} heatmap");
        hChart.xAxis().title("r, Å");
        hChart.yAxis().title("time, ps (window center)");
        hChart.colorScale(anychart.scales.linearColor());
        hChart.container("heatmap_chart");
        hChart.draw();
      }})();

      // RDF в скользящем окне
      var selectEl = document.getElementById("window_select");
      if (!windowSeries || windowSeries.length === 0) {{
        document.getElementById("window_chart").innerText = "No windowed RDF data for {pair_label}.";
      }} else {{

        // наполняем select
        windowSeries.forEach(function(w, idx) {{
          var opt = document.createElement("option");
          var t = (w.center_time !== undefined && !isNaN(w.center_time)) ? w.center_time.toFixed(3) : ("#" + idx);
          opt.value = idx;
          opt.text = "Window " + idx + " (t ≈ " + t + " ps)";
          selectEl.appendChild(opt);
        }});

        var wChart = anychart.line();
        wChart.title("RDF {pair_label} – sliding window (10% length, step 2%)");
        wChart.xAxis().title("r, Å");
        wChart.yAxis().title("g(r)");
        var wSeriesLine = wChart.line([]);
        wSeriesLine.name("g_{pair_label} (window)");
        wChart.container("window_chart");
        wChart.draw();

        function updateWindow(idx) {{
          var w = windowSeries[idx];
          if (!w || !w.series) return;
          var data = w.series.map(function(p) {{
            return {{ x: p.r, value: p.g }};
          }});
          wSeriesLine.data(data);
        }}

        selectEl.addEventListener("change", function() {{
          var idx = parseInt(this.value, 10);
          if (!isNaN(idx)) {{
            updateWindow(idx);
          }}
        }});

        // первый
        updateWindow(0);
        selectEl.value = "0";
      }}

      // График с доверительными интервалами
      (function() {{
        if (!windowSeries || windowSeries.length === 0) {{
          document.getElementById("ci_chart").innerText = "No windowed RDF data for {pair_label}.";
          return;
        }}

        // Сбор всех r-бинов
        var rSet = {{}};
        windowSeries.forEach(function(w) {{
          if (!w || !w.series) return;
          w.series.forEach(function(p) {{
            if (p.r === undefined || p.g === undefined) return;
            rSet[p.r] = true;
          }});
        }});

        var rValues = Object.keys(rSet).map(parseFloat).sort(function(a, b) {{ return a - b; }});
        if (rValues.length === 0) {{
          document.getElementById("ci_chart").innerText = "No windowed RDF data for {pair_label}.";
          return;
        }}

        // Для каждого r считаем mean и 95% CI по окнам
        var meanData = [];
        var lowerData = [];
        var upperData = [];

        rValues.forEach(function(rVal) {{
          var vals = [];
          windowSeries.forEach(function(w) {{
            if (!w || !w.series) return;
            for (var i = 0; i < w.series.length; i++) {{
              var p = w.series[i];
              if (Math.abs(p.r - rVal) < 1e-6 && !isNaN(p.g)) {{
                vals.push(p.g);
                break;
              }}
            }}
          }});
          if (vals.length === 0) return;

          var n = vals.length;
          var sum = 0.0;
          for (var i = 0; i < n; i++) sum += vals[i];
          var mean = sum / n;
          var varSum = 0.0;
          for (var i = 0; i < n; i++) {{
            var diff = vals[i] - mean;
            varSum += diff * diff;
          }}
          var std = n > 1 ? Math.sqrt(varSum / (n - 1)) : 0.0;
          var se = n > 0 ? std / Math.sqrt(n) : 0.0;
          var ci = 1.96 * se;

          meanData.push({{ x: rVal, value: mean }});
          lowerData.push({{ x: rVal, value: mean - ci }});
          upperData.push({{ x: rVal, value: mean + ci }});
        }});

        if (meanData.length === 0) {{
          document.getElementById("ci_chart").innerText = "Not enough data for confidence intervals.";
          return;
        }}

        var ciChart = anychart.line();
        ciChart.title("RDF {pair_label} – mean ± 95% CI over windows");
        ciChart.xAxis().title("r, Å");
        ciChart.yAxis().title("g(r)");

        var meanSeries = ciChart.line(meanData);
        meanSeries.name("mean g_{pair_label}");
        meanSeries.stroke("#1f77b4", 2);

        // Рисуем доверительный интервал как две линии
        var lowerSeries = ciChart.line(lowerData);
        lowerSeries.name("lower 95% CI");
        lowerSeries.stroke("#aec7e8", 1, "10 5", "round");

        var upperSeries = ciChart.line(upperData);
        upperSeries.name("upper 95% CI");
        upperSeries.stroke("#aec7e8", 1, "10 5", "round");

        ciChart.legend(true);
        ciChart.container("ci_chart");
        ciChart.draw();
      }})();
    }});
  </script>
</body>
</html>
"""
    return html


def generate_msd_html(
    msd_series: dict,
    msd_windows: dict,
) -> str:
    """
    Страница с MSD по типам молекул:
    - верх: полное MSD(t) по каждому типу
    - низ: MSD с усреднением по скользящему окну (10%, шаг 2%) – по выбранному типу
    """
    types = sorted(msd_series.keys())

    msd_json = json.dumps(msd_series)
    msd_win_json = json.dumps(msd_windows)
    types_json = json.dumps(types)

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>MSD by molecule types</title>
  <script src="https://cdn.anychart.com/releases/8.14.1/js/anychart-base.min.js"></script>
  <style>
    body {{
      font-family: sans-serif;
      margin: 0;
      padding: 0;
    }}
    #msd_full {{
      width: 100%;
      height: 500px;
    }}
    #msd_win {{
      width: 100%;
      height: 500px;
    }}
    #msd_controls {{
      padding: 10px;
    }}
    a.back {{
      display: inline-block;
      margin: 10px;
    }}
  </style>
</head>
<body>
  <a href="index.html" class="back">&larr; Back to overview</a>
  <h1 style="padding: 0 10px 10px 10px;">MSD by molecule types</h1>

  <div id="msd_full"></div>

  <div id="msd_controls">
    <label for="type_select">Molecule type: </label>
    <select id="type_select"></select>
  </div>
  <div id="msd_win"></div>

  <script>
    var msdSeries = {msd_json};
    var msdWindows = {msd_win_json};
    var types = {types_json};

    anychart.onDocumentReady(function() {{
      // Полное MSD(t)
      var fullChart = anychart.line();
      fullChart.title("MSD(t) per molecule type");
      fullChart.xAxis().title("time, ps");
      fullChart.yAxis().title("MSD, Å^2");

      types.forEach(function(name) {{
        var data = (msdSeries[name] || []).map(function(p) {{
          return {{ x: p.time, value: p.msd }};
        }});
        var s = fullChart.line(data);
        s.name(name);
      }});

      fullChart.legend(true);
      fullChart.container("msd_full");
      fullChart.draw();

      // Скользящее окно: усреднённое MSD
      var typeSelect = document.getElementById("type_select");
      types.forEach(function(name) {{
        var opt = document.createElement("option");
        opt.value = name;
        opt.text = name;
        typeSelect.appendChild(opt);
      }});

      var winChart = anychart.line();
      winChart.title("MSD – sliding window (10% length, step 2%)");
      winChart.xAxis().title("time, ps (window center)");
      winChart.yAxis().title("MSD, Å^2 (window-averaged)");

      var winSeries = winChart.line([]);
      winSeries.name("MSD (window)");

      winChart.container("msd_win");
      winChart.draw();

      function updateWin(typeName) {{
        var ws = msdWindows[typeName] || [];
        var data = ws.map(function(p) {{
          return {{ x: p.center_time, value: p.msd }};
        }});
        winSeries.data(data);
      }}

      typeSelect.addEventListener("change", function() {{
        updateWin(this.value);
      }});

      if (types.length > 0) {{
        typeSelect.value = types[0];
        updateWin(types[0]);
      }} else {{
        document.getElementById("msd_win").innerText = "No MSD data.";
      }}
    }});
  </script>
</body>
</html>
"""
    return html


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Чтение архива MD‑запуска и построение HTML‑отчёта с графиками (AnyChart)."
    )
    ap.add_argument("archive", type=Path, help="tar.gz архив с результатами запуска.")
    ap.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Директория для отчёта. По умолчанию рядом с архивом с суффиксом _report.",
    )

    args = ap.parse_args()

    # каталог отчёта по умолчанию: <archive_name>_report/
    if args.output_dir is None:
        base = args.archive
        if base.suffix in {".gz", ".bz2", ".xz"}:
            base = base.with_suffix("")
        if base.suffix == ".tar":
            base = base.with_suffix("")
        args.output_dir = base.parent / f"{base.name}_report"
    report_dir: Path = args.output_dir
    report_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        with tarfile.open(args.archive, "r:gz") as tar:
            tar.extractall(tmpdir)

        entries = list(tmpdir.iterdir())
        if not entries:
            raise SystemExit("Архив пуст.")
        root = entries[0] if len(entries) == 1 else tmpdir

        energy_path = find_file(root, "energy.csv")
        hbonds_path = find_file(root, "hbonds.csv")
        rdf_path = find_file(root, "rdf.csv")
        sys_path = find_file(root, "system_params.csv")

        energy = read_csv(energy_path) if energy_path else None
        hbonds = read_csv(hbonds_path) if hbonds_path else None
        rdf = read_csv(rdf_path) if rdf_path else None
        sys_params = read_csv(sys_path) if sys_path else None

        # index.html
        index_html = generate_index_html(energy, hbonds, rdf, sys_params)
        (report_dir / "index.html").write_text(index_html, encoding="utf-8")

        # MSD по молекулам (если есть mapping и xtb.trj)
        msd_series = {}
        msd_windows = {}
        mapping = load_mapping(root)
        if mapping is not None:
            trj_path2 = find_file(root, "xtb.trj")
            if trj_path2:
                # Определяем dump_fс тем же способом, что и выше
                coord_path2 = find_file(root, "coord")
                dump_fs2 = None
                if coord_path2 and coord_path2.is_file():
                    text2 = coord_path2.read_text(encoding="utf-8", errors="ignore")
                    for line in text2.splitlines():
                        line_stripped = line.strip()
                        if line_stripped.startswith("dump"):
                            try:
                                after_eq = line_stripped.split("=", 1)[1]
                                dump_fs2 = float(after_eq.split()[0])
                            except Exception:
                                pass
                            break
                if dump_fs2 is None:
                    dump_fs2 = 1.0

                frames_meta2, coords_list2, _ = parse_xtb_trj(trj_path2, dump_fs2)
                msd_series, msd_windows = compute_msd_per_type(coords_list2, frames_meta2, mapping)

        # msd.html
        if msd_series:
            msd_html = generate_msd_html(msd_series, msd_windows)
            (report_dir / "msd.html").write_text(msd_html, encoding="utf-8")

        # Попробуем рассчитать оконные RDF по xtb.trj + coord
        window_rdfs: dict[str, List[dict[str, Any]]] = {
            "g_OH": [],
            "g_OO": [],
            "g_NH": [],
            "g_NO": [],
            "g_NN": [],
        }
        trj_path = find_file(root, "xtb.trj")
        coord_path = find_file(root, "coord")
        if trj_path and coord_path:
            # извлекаем dump из coord ($md блок)
            dump_fs = None
            text = coord_path.read_text(encoding="utf-8", errors="ignore")
            for line in text.splitlines():
                line_stripped = line.strip()
                if line_stripped.startswith("dump"):
                    # ожидается формат: dump= X  # in fs
                    try:
                        after_eq = line_stripped.split("=", 1)[1]
                        dump_fs = float(after_eq.split()[0])
                    except Exception:
                        pass
                    break
            if dump_fs is None:
                dump_fs = 1.0

            frames_meta, coords_list, symbols_list = parse_xtb_trj(trj_path, dump_fs)
            n_frames = len(coords_list)
            if n_frames > 0:
                win_len = max(1, int(0.10 * n_frames))
                step = max(1, int(0.02 * n_frames))
                if win_len > n_frames:
                    win_len = n_frames

                box = get_box_from_coord(coord_path)

                pair_defs = [
                    ("g_OH", ("O", "H")),
                    ("g_OO", ("O", "O")),
                    ("g_NH", ("N", "H")),
                    ("g_NO", ("N", "O")),
                    ("g_NN", ("N", "N")),
                ]

                for start in range(0, n_frames - win_len + 1, step):
                    end = start + win_len
                    sub_coords = coords_list[start:end]
                    sub_symbols = symbols_list[start:end]
                    # время окна – середина
                    center_idx = start + win_len // 2
                    center_time = frames_meta[center_idx]["time_ps"]

                    for key, (a, b) in pair_defs:
                        try:
                            r_arr, g_arr = compute_rdf_pair(
                                sub_coords,
                                sub_symbols,
                                a,
                                b,
                                r_max=10.0,
                                dr=0.1,
                                box=box,
                            )
                        except Exception:
                            continue
                        series = [
                            {"r": float(r_arr[i]), "g": float(g_arr[i])}
                            for i in range(len(r_arr))
                        ]
                        window_rdfs[key].append(
                            {
                                "center_time": float(center_time),
                                "series": series,
                            }
                        )

        # RDF per pair
        pairs = [
            ("g_OH", "OH", "rdf_OH.html"),
            ("g_OO", "OO", "rdf_OO.html"),
            ("g_NH", "NH", "rdf_NH.html"),
            ("g_NO", "NO", "rdf_NO.html"),
            ("g_NN", "NN", "rdf_NN.html"),
        ]
        for key, label, fname in pairs:
            html = generate_rdf_pair_html(key, label, rdf, window_rdfs.get(key))
            (report_dir / fname).write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()


