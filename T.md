# Streamlit + Python — 3-Hour Class Transcript

A speakable script for a three-hour hands-on class built around the two demos
in this repo. Portwatch goes first because it's the dashboard shape students
will actually build at work; solar comes second as the "and now here's what
you do when the data doesn't fit in a CSV" capstone.

- `portwatch/app.py` — operational dashboard for daily ship transits through
  the Strait of Hormuz, using (synthetic by default) IMF PortWatch data.
  Sidebar filters, KPIs, time series, stacked breakdown, click-to-drill-down,
  geo map.
- `solar/app.py` — a map of where the Sun, Moon and planets project down onto
  the Earth for any given instant. Backed by a sharded binary ephemeris
  dataset of ~372 monthly files covering 2000–2030.

Focus themes: **context filters** (Hour 2, on portwatch) and **managing large
data loads** (Hour 3, introducing solar).

Lines in plain text are what you *say*. Lines in `> blockquotes` are stage
directions — what to click, what to run, what to draw on the board. Code
blocks are what students should have on screen.

---

## Before class (5 min before students arrive)

```bash
cd streamlit-demos
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python portwatch/scripts/generate_sample_data.py
streamlit run portwatch/app.py
```

Have portwatch running in terminal 1 and ready to open solar in terminal 2.
Have the repo open in your editor with `portwatch/app.py` visible. Browser
with two tabs ready, one per app. A blank scratch Python REPL for the
timing demos in Hour 3.

---

# HOUR 1 — What Streamlit actually is (0:00 – 1:00)

## 0:00 — Opening (5 min)

Welcome. Three hours, two real apps, one big idea: **Streamlit turns a
Python script into a web app, and the entire framework is built around the
fact that your script reruns top-to-bottom every time the user clicks
anything.**

That sentence is the entire mental model. If you leave today with only
that, you'll be fine. Every other Streamlit concept — widgets, caching,
context filters, session state — falls out of that one rule.

We're going to spend the first hour getting that rule into your fingers
using a dashboard I want you to imagine landing in your inbox on a Monday
morning: "we need a dashboard for daily ship transits through the Strait
of Hormuz, with filters and drill-down, by Friday." Realistic shape.
Realistic constraints. By the end of the hour you'll understand the whole
280-line file.

Hour two is context filters — how a sidebar control or a click on a chart
reshapes the rest of the page. Hour three is what happens when your data
is too big to read with `pd.read_csv` — and for that we switch to the
solar demo, which loads 30 years of planetary positions from sharded
binary files.

> **Draw on the board:**
> ```
>     widget change ──▶ rerun script top to bottom ──▶ new output
> ```
> Leave that up for the whole session. You will point at it a lot.

Questions welcome any time. If something looks like magic, stop me —
Streamlit has very few genuine magic parts.

## 0:05 — What is Streamlit, really? (5 min)

Streamlit is a Python library. That's it. You `pip install streamlit`,
you write a Python script that calls functions like `st.title(...)` and
`st.slider(...)`, and you run it with `streamlit run app.py`. Streamlit
spins up a little web server and a browser tab, and renders whatever
your script produces.

The framework was designed for data people who know Python and pandas
but don't want to learn React or Flask. The trade-off: it's opinionated,
and not as flexible as a real web framework. For "I have a script, I
want my colleagues to poke at it," the trade-off is almost always worth
it.

Two things make it feel different from notebooks:

1. **It's a real web app.** You send a URL to someone. No Jupyter
   kernel on their machine.
2. **Widgets are live.** Changing a slider reruns your script and
   updates the output automatically. You do not write callback code.

That second point is where the weirdness lives. Let's look at it.

## 0:10 — First look at portwatch (10 min)

> **Action:** switch to the browser tab running `portwatch/app.py`. Default
> view should be the last 180 days of the Strait of Hormuz dataset.

Take this in. Title at top, a four-metric KPI row showing latest day, 7-day
average, 30-day average, year-ago comparison. Below that, a time-series
chart with a rolling average line. Below that, a stacked bar chart broken
out by ship type — tankers in red, dry bulk in brown, containers in blue.
Below *that*, a drill-down placeholder. Below *that*, a map of the
Strait of Hormuz with a marker sized by transit volume. And a sidebar on
the left with date range, ship type multiselect, and a rolling-average
slider.

> **Action:** shrink the date range to the last 30 days. The KPIs update,
> the charts redraw. Deselect "tanker" from the multiselect. The red
> segments vanish from the stacked bar. Move the rolling average slider
> from 7 to 21. The smooth line gets smoother.

Three separate interactions, three full page updates, zero JavaScript.
And the thing I want you to hold in your head: **every one of those
interactions reran the entire Python script from line 1 to the end.**

> **Action:** click a point on the daily time-series. A new drill-down
> chart appears below showing the breakdown by cargo type for that day.

That's a *context filter* — a click in one chart reshaping another. That
is the headline feature for Hour 2.

Now the source. The whole file is about 280 lines. That's not rhetoric,
that's the entire app.

> **Action:** open `portwatch/app.py` in the editor. Scroll from top to
> bottom once without narrating. Let students see the full shape. Then
> back to the top.

## 0:20 — Reading `portwatch/app.py` top to bottom (15 min)

Lines 1-16: imports. Normal Python. The only Streamlit thing is
`import streamlit as st`.

Lines 18-28: constants. Data path. Hormuz coordinates for the map. A
color map keyed by ship type — tanker red, container blue, and so on —
so the stacked bar and the drill-down share the same palette.

Lines 31-43 — the first Streamlit-specific thing:

```python
@st.cache_data(ttl=3600, show_spinner=False)
def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        st.error(...)
        st.stop()
    df = pd.read_csv(path, parse_dates=["date"])
    df = df[df["chokepoint_name"].str.contains("Hormuz", case=False, na=False)]
    return df
```

Hold onto `@st.cache_data`. We'll come back to it in Hour 3 — it's the
single most important decorator in Streamlit. For now, note what it
*does*: it memoises the return value of this function. The first time
we call `load_data(path)`, we actually read the CSV. Subsequent calls
with the same path hit the cache and return instantly. `ttl=3600` means
the cache expires after an hour, so a file that gets refreshed daily
is picked up eventually.

Two more things in that function. `st.error` renders a red error box.
`st.stop()` halts the script — like `return` but for the whole
Streamlit run, not just the function. If the CSV isn't there, we show
the error and bail out before anything downstream crashes.

Line 46 is the first line that renders UI:

```python
st.set_page_config(page_title="Hormuz Transit Monitor", layout="wide")
```

`set_page_config` must come before any other `st.` call. It sets the
browser tab title and tells Streamlit to use a wide layout. Call it
second and you get an error.

Lines 47-51: title and caption. `st.caption` is just smaller gray
text. `st.title` is H1.

Line 53: we call `load_data`. Cached. Returns the full DataFrame
filtered to Hormuz rows.

Lines 55-78 is the sidebar block. This is where every filter lives.
I'm going to walk you through this in detail in Hour 2, but I want you
to notice the pattern right now:

```python
st.sidebar.header("Filters")
date_range = st.sidebar.date_input(...)
selected_types = st.sidebar.multiselect(...)
rolling_window = st.sidebar.slider(...)
```

Each of these returns the current value. Not a callback — the *value*.
`date_range` is a tuple of two `datetime.date` objects. `selected_types`
is a list of strings. `rolling_window` is an int. Plain Python, ready
to use.

Lines 80-86 — this is the pattern I want everyone to internalise:

```python
mask = (
    (df.date.dt.date >= start)
    & (df.date.dt.date <= end)
    & (df.cargo_type.isin(selected_types))
)
fdf = df.loc[mask].copy()
```

**Filter once, upstream of every view.** We build one mask from the
sidebar values. We apply it once to the full DataFrame. We get `fdf`
— "filtered df" — and from here to the bottom of the file, every KPI,
every chart, the drill-down, the map, the raw rows table, all of them
read from `fdf`. Nothing further down filters from scratch.

Why this matters: if you filter inside every view function, you're
doing the same work three or four times per rerun. On a big DataFrame
that's visible in the UI as lag. One filter, reused everywhere.

Lines 88-90 is the empty-result guard:

```python
if fdf.empty:
    st.warning("No rows match the current filters.")
    st.stop()
```

If the user somehow picks a combination with no data, we show a
warning and `st.stop()`. Everything downstream assumes `fdf` is
non-empty. This one line saves you from a dozen `IndexError` tracebacks.

Lines 93-125 is the KPI block. Group by date, sum transits, compute
the latest day's value, 7-day average, 30-day average, year-ago
comparison. Then `st.metric` — which is the four cards at the top of
the page. Notice the `delta=` argument; that's what gives you the
little green-up or red-down arrow. `st.columns(4)` splits the row
into four equal cells and we drop one metric into each.

Lines 127-152 is the time-series chart — we'll come back to this
when we do context filters, because lines 146-152 are where the
magic happens. For now: we build a Plotly figure, we call
`st.plotly_chart(fig)`, it renders.

Lines 154-170: the stacked bar. Same pattern. Plotly figure,
`st.plotly_chart`.

Lines 172-207: the drill-down — that's Hour 2 material.

Lines 209-263: the geo map — Plotly `Scattergeo`. Note the map is
sized by `latest_val`, so as you change the date range, the marker
changes size.

Lines 265-266: the raw-rows expander at the bottom.

```python
with st.expander("Raw rows (latest 50)"):
    st.dataframe(fdf.sort_values("date").tail(50), use_container_width=True)
```

`st.expander` creates a collapsible section. Good for "I want the
data available but not in the way." `st.dataframe` is an interactive
table — sortable columns, filter on hover, copy to clipboard. You
get that for free.

And now — the key thing. On every single widget interaction —
every tick of the date range, every toggle of a ship type, every
drag of the rolling window slider — *this entire file runs again*
from line 1 to line 266. `load_data` is cached so it's cheap. The
mask is rebuilt. `fdf` is filtered again. All the Plotly figures
are built again. Every `st.plotly_chart` call renders again.

> **Question to ask students:** "If the whole script reruns on every
> widget change, what could go wrong as the app grows?"
>
> Collect answers: "it'll be slow", "I'll reload data every time",
> "my API calls will repeat." Good. That's Hour 3.

## 0:35 — Prove it: the rerun probe (5 min)

Let's prove the script is really rerunning.

> **Action:** in `portwatch/app.py`, right after `st.title(...)`, add:
>
> ```python
> import time as _t
> st.caption(f"Rendered at {_t.strftime('%H:%M:%S')}")
> ```
>
> Save. The app hot-reloads. Move the date range. Timestamp changes.
> Toggle a ship type. Timestamp changes. Drag the slider. Timestamp
> changes.

Every interaction reran the whole script. If you came from Flask you
are now mildly horrified. Yes — we rerun the full view function on
every click. That's why caching is a first-class citizen here, and
it's why we `@st.cache_data` anything expensive.

> **Action:** remove the probe line before moving on.

## 0:40 — Widgets tour on a scratch file (10 min)

I'm going to show the widgets you'll actually use on a blank file
rather than touching the real app.

> **Action:** create `scratch.py` in the repo root:
>
> ```python
> import streamlit as st
>
> st.title("Widget zoo")
>
> name = st.text_input("Your name", value="world")
> st.write(f"Hello, {name}")
>
> age = st.slider("Age", 0, 120, 30)
> st.write(f"Age: {age}")
>
> color = st.selectbox("Favourite colour", ["red", "green", "blue"])
> st.write(f"Colour: {color}")
>
> toppings = st.multiselect("Pizza", ["ham", "cheese", "olives", "mushroom"])
> st.write(f"{len(toppings)} toppings")
>
> agree = st.checkbox("I agree")
> if agree:
>     st.success("Great.")
>
> go = st.button("Go")
> if go:
>     st.balloons()
> ```
>
> Run it: `streamlit run scratch.py`. Poke every widget.

The pattern is always the same. The widget function *returns the
current value*. You write `if agree:` — that's a normal Python
boolean. No callbacks, no observers. Just values and `if`s.

One thing about buttons: `st.button("Go")` returns `True` exactly
once, on the rerun caused by the click itself. If the user then
moves a slider, the script reruns, and `go` is now `False`, because
this rerun wasn't caused by the button. This trips up everyone the
first time. If you need to *remember* that a button was clicked,
that's `st.session_state`, which we hit in Hour 2.

## 0:50 — Layout: columns, sidebar, tabs, expanders (10 min)

Four layout tools cover 95% of what you need.

**Columns.** `st.columns(n)` or `st.columns([ratios])`. You saw
this in the KPI row at `portwatch/app.py:110` —
`k1, k2, k3, k4 = st.columns(4)`. Put widgets in a column by
calling the method on the column object: `k1.metric(...)` instead
of `st.metric(...)`.

**Sidebar.** `st.sidebar.something(...)` puts the widget in the
left sidebar instead of the main flow. Convention: filters in the
sidebar, results in the main area. Portwatch follows this
religiously — every filter control is under `st.sidebar`.

**Tabs.** `tab1, tab2 = st.tabs(["Overview", "Details"])`, then
`with tab1: ...`. Good for alternative views of the same data.

**Expanders.** `with st.expander("Raw data"): st.dataframe(df)`.
A collapsible section. Both apps use this for raw rows at the
bottom.

> **Action:** extend `scratch.py`:
>
> ```python
> with st.sidebar:
>     st.header("Sidebar filters")
>     n = st.slider("How many?", 1, 100, 10)
>
> tab1, tab2 = st.tabs(["First", "Second"])
> with tab1:
>     st.write(f"You picked {n}")
> with tab2:
>     with st.expander("Hidden details"):
>         st.write("Surprise.")
> ```

One thing about layout: you do not get pixel-perfect control. If
you want floating panels with draggable resize handles, that's not
Streamlit and trying to force it will make you miserable. Streamlit
is top-to-bottom flow with some structure on top. Accept that and
you'll move fast.

## 1:00 — First break (5 min)

> **5-minute break.** Check the app still runs.

---

# HOUR 2 — Context filters (1:05 – 2:05)

## 1:05 — What "context filter" means (5 min)

A context filter is: **a selection in one place narrows down what I
see somewhere else on the page.** Every BI tool has them. Tableau
calls them this name. The value is that once a user has drilled into
something, everything else reflects that choice without them having
to repeat it.

Portwatch has two flavours, and we're going to build on both:

1. **Global filters** — sidebar controls that constrain every view
   at once. Date range, ship type, rolling window. You already saw
   the `fdf = df.loc[mask].copy()` line — that's where the global
   filters land.
2. **Cross-chart filters** — click a point in one chart, another
   chart reacts. Until Streamlit 1.28 this was genuinely hard to
   do. Now it's five lines.

We're going to look at both, and then you're going to build one.

## 1:10 — Global filters, in depth (10 min)

Back to `portwatch/app.py`, line 55.

```python
st.sidebar.header("Filters")
min_d, max_d = df.date.min().date(), df.date.max().date()
default_start = max(min_d, max_d - timedelta(days=180))
date_range = st.sidebar.date_input(
    "Date range",
    value=(default_start, max_d),
    min_value=min_d,
    max_value=max_d,
)
```

Three things worth calling out.

**One:** the date picker's bounds come from the *data*, not
hard-coded. If you drop a different CSV in, the picker adapts
automatically. This is a habit worth cultivating — read your
sensible defaults from the data whenever you can.

**Two:** the default start is "180 days before the most recent
date." Not `date.today()`. Data dashboards often have stale data,
and if you default to "today" the user opens the app and sees an
empty chart and assumes the app is broken. Default to the *most
recent day in the data*, and work backward from there.

**Three:** passing `value=(start, end)` as a tuple turns
`date_input` into a *range* picker. Lesser-known but essential.

Lines 66-69:

```python
if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = date_range
else:
    start, end = default_start, max_d
```

This guards a quirk. `st.date_input` can briefly return a single
date if the user has picked a new start but not yet the new end.
Without this guard, you get a crash on the in-between rerun.
Remember: the script reruns on *every* click, including the click
where the state is half-updated.

Lines 71-76: ship type multiselect. Note the default is *all*
types selected — otherwise the first render shows an empty page,
and users bounce.

Line 78: `rolling_window = st.sidebar.slider("Rolling average (days)", 1, 30, 7)`.
This one doesn't filter rows — it parameterises a *computation*
downstream (the rolling mean on the time series). Same control
shape; different effect.

Lines 80-86: the mask and `fdf`. We already saw this. One filter,
reused by everything downstream.

> **Demo:** in the browser, move the date range. Watch all four
> KPIs, both charts, and the map update in lockstep. Deselect
> "container" from the multiselect. Watch the blue go out of the
> stacked bar. Push the rolling window slider from 7 to 30. Watch
> only the smooth line change — the raw "Daily" line is untouched.

Four views, one pipeline, no duplicated filter logic. That's the
template for every dashboard you'll ever build in this framework.

## 1:20 — The click-to-drill-down pattern (15 min)

Now the fun one. Scroll to line 146:

```python
ts_event = st.plotly_chart(
    fig_ts,
    use_container_width=True,
    on_select="rerun",
    selection_mode="points",
    key="ts_chart",
)
```

Look at the new arguments. `on_select="rerun"` tells Streamlit:
"when the user selects something inside this chart, rerun the
script." `selection_mode="points"` says 'selecting' means clicking
individual points. Other modes: `"box"` and `"lasso"` for
rectangular or freeform selection.

`key="ts_chart"` gives the widget a stable identity — important
if you have multiple similar charts on the same page.

And the kicker: `st.plotly_chart` used to return `None`. With
`on_select="rerun"`, it returns an *event object* you can read.
That's `ts_event`.

Scroll down to line 175:

```python
picked_date = None
try:
    points = ts_event["selection"]["points"]
    if points:
        picked_date = pd.Timestamp(points[0]["x"]).normalize()
except (KeyError, TypeError, IndexError):
    picked_date = None
```

We read the event. If the user has clicked a point, `points` is
a non-empty list and the first point's `x` is the date they
clicked. If not — if this rerun was triggered by a sidebar change
instead — `points` is empty and `picked_date` stays `None`.

The `try/except` is defensive: the schema of the event object
isn't guaranteed across Streamlit versions, and you don't want a
`KeyError` blowing up the whole page if a point's payload shape
shifts.

Lines 183-207:

```python
if picked_date is None:
    st.caption("Click a point on the daily time-series above to drill into that day.")
else:
    day_rows = fdf[fdf.date.dt.normalize() == picked_date]
    ...
    fig_day = px.bar(breakdown, x="cargo_type", y="transits", ...)
    st.plotly_chart(fig_day, use_container_width=True)
```

Without a click: hint. With a click: filter `fdf` down to just
that day, aggregate by cargo type, render a new chart.

Here's the insight. **There is no event listener.** No onClick
callback. No observer wiring. The entire interaction is:

1. User clicks a point.
2. Streamlit reruns the script.
3. This time, `ts_event` contains the clicked point.
4. The `if picked_date is not None:` branch takes a different
   path, and renders a different chart.

It's the same reactive model as every other widget. The event
object is just another "what's the current state" input — the same
kind of input `st.slider` gives you.

> **Demo:** click random points on the daily line. Drill-down
> chart appears and updates. Click again. Click off the line to
> deselect — drill-down reverts to the hint.

If you've ever tried to do this in raw Plotly Dash or custom JS
you know what a six-line miracle this is.

## 1:35 — Hands-on: add a second context filter (20 min)

Now you're going to add one. The stacked bar chart at line 170 is
currently static — clicking it does nothing. I want you to make
clicking a segment filter the *map* at the bottom by cargo type.

The steps:

1. Add `on_select`, `selection_mode`, and `key` to the
   `st.plotly_chart` call for `fig_bar` at line 170, and capture
   the return:

   ```python
   bar_event = st.plotly_chart(
       fig_bar,
       use_container_width=True,
       on_select="rerun",
       selection_mode="points",
       key="bar_chart",
   )
   ```

2. After reading `ts_event`, read `bar_event` similarly:

   ```python
   picked_type = None
   try:
       bar_points = bar_event["selection"]["points"]
       if bar_points:
           picked_type = bar_points[0].get("customdata")
   except (KeyError, TypeError, IndexError):
       picked_type = None
   ```

3. For that `customdata` lookup to have a value, add
   `custom_data=["cargo_type"]` to the `px.bar(...)` call that
   builds `fig_bar`. That's how you attach arbitrary payload to
   each point so it comes back in the click event.

4. In the map section (from line 210), if `picked_type` is set,
   filter `fdf` down to that type before computing
   `total_transits` and the marker size:

   ```python
   map_df = fdf if picked_type is None else fdf[fdf.cargo_type == picked_type]
   ```

   and use `map_df` instead of `fdf` in the total/marker math.
   Update the caption so the user knows the map is filtered.

This is a realistic exercise — it involves reading Plotly's event
schema, wiring `customdata`, and deciding what to do when there's
no click. Don't skip the no-click branch. Making filters clearable
is half the job.

> **Do this live for ~8 minutes, then let students try for ~12.**
> Walk around. Most will get stuck on the `customdata` step —
> that's the point.

## 1:55 — `st.session_state` (5 min)

One last thing before break. You'll eventually want to remember
something *across* widgets. For example: "user clicks a date in
the time series, I want to remember it even after they drag the
sidebar date range."

That's `st.session_state`. A dictionary that lives for the
duration of one user's session and survives reruns.

```python
if picked_date is not None:
    st.session_state["drill_date"] = picked_date

if "drill_date" in st.session_state:
    day_rows = fdf[fdf.date.dt.normalize() == st.session_state["drill_date"]]
    ...

if st.button("Clear selection"):
    st.session_state.pop("drill_date", None)
```

Rule of thumb: use `session_state` when you need state *widgets
don't give you for free*. If a widget is on the page and returns
the value you want, you don't need `session_state`. If you need to
remember something across a navigation or across widgets, you do.

## 2:00 — Second break (5 min)

> **5-minute break.** Switch the running app: stop portwatch, start
> solar. `streamlit run solar/app.py`.

---

# HOUR 3 — Managing large data loads (2:05 – 3:05)

## 2:05 — The problem, stated bluntly (5 min)

Here's the problem we've been dancing around. The script reruns on
every interaction. So if your script starts with:

```python
df = pd.read_csv("huge.csv")
```

…then every time the user moves a slider, you reread a huge CSV.
For a 500 MB file that might be three seconds. Three seconds per
click. The app is unusable.

There's also a second version. If you build some heavy *object* —
a machine learning model, a database connection pool, or, as you're
about to see, an ephemeris loader that opens 372 binary files — you
do not want to rebuild it on every rerun either. You want to build
it once and share it across reruns and across users hitting the
same server.

Streamlit has two decorators, one for each case. They look similar.
They are not the same. Mixing them up is the single most common
source of bugs in Streamlit apps, and I want you to leave today
knowing exactly which one to reach for.

And to show off the second one, we're switching demos.

## 2:10 — Meet the solar demo (5 min)

> **Action:** browser tab two. Default view is June 21, 2024, with
> "Day / night" overlay. Map of the world, Sun, Moon, planets
> projected onto the Earth's surface, terminator line between day
> and night.

This app answers: *for any instant in time, where on Earth is the
Sun directly overhead? Where's the Moon? Where's Jupiter?* The math
is in `solar/projection.py` and `solar/ephemeris.py`; we're not
touching those files today — treat them as a library. What I care
about is the data layer.

> **Action:** in a terminal:
> ```
> ls solar/data | wc -l
> du -sh solar/data
> ```
> Students see: ~373 files, tens of megabytes.

One file per month, from January 2000 through December 2030. Each
file holds *hourly* positions of ten solar system bodies — about
744 rows per month, 43 fields per row, 172 bytes per row. 46 MB
total on disk. Not enormous, but big enough that you do NOT want
to rehydrate it on every slider drag.

And here's the interesting thing I want you to sit with for a
second. **How much of that 46 MB do you think this app actually
loads into memory at any one time?**

> Let students guess. Expect "most of it" / "half of it" / "all of
> it." Then drop the answer.

**128 kilobytes.** One month. 0.3% of the dataset. For the default
view — one instant in time — the loader reads exactly one monthly
file. If you switch to the Annual sun path overlay, it reads the
twelve months of that year — about 1.5 MB, or 3.3% of the total.
And there's a hard cap: the loader's internal chunk cache evicts
anything beyond the 12 most-recently-used months, so the process
footprint never grows past ~1.5 MB no matter how long the app
runs.

Three layers make this work, and we're about to meet all three.
A shared loader object behind `@st.cache_resource`. The loader's
own internal dict-cache for recent months. And `@st.cache_data`
on derived views like the annual sun path. Each layer caches at
the right level.

## 2:15 — `@st.cache_data` vs `@st.cache_resource` (15 min)

Open `solar/app.py`. Line 54:

```python
@st.cache_resource
def get_loader() -> EphemerisLoader:
    return EphemerisLoader(DATA_DIR)
```

And line 59:

```python
@st.cache_data(show_spinner=False)
def annual_sun_path(_loader: EphemerisLoader, year: int):
    ...
```

Two decorators. One builds an object. The other returns data. Let's
take them one at a time.

**`@st.cache_resource`** memoises *an object*. It returns the exact
same underlying Python object to every caller. It doesn't serialise
and it doesn't copy. Use it for things that are expensive to
construct and should be shared: database connections, ML model
weights, file handles, our ephemeris loader.

Here, `EphemerisLoader` opens file handles, holds a memory-mapped
view of recently-read months, and has internal state you want
shared across every user and every rerun. You absolutely do not
want to rebuild it on every slider drag. `cache_resource` guarantees
there is **one** loader per server process.

**`@st.cache_data`** memoises *a value*. It hashes the arguments,
and if it has a cached result for that hash it returns it without
running the function. It serialises the result, which means each
caller gets their own copy — safe, isolated. Good for DataFrames,
lists, dicts, numpy arrays.

`annual_sun_path(_loader, year)` computes a list of 365 subsolar
points for a given year. Each year takes a few hundred ms to
compute. After the first call with `year=2024`, any subsequent
call with the same year is instant.

> **The rule to memorise:**
>
> - Return value is **data** (DataFrame, dict, list, array)? →
>   `@st.cache_data`
> - Return value is a **thing** (connection, loader, model)? →
>   `@st.cache_resource`
>
> Data gets copied per caller. Resources are shared singletons.

Portwatch also uses `@st.cache_data` — go back to
`portwatch/app.py:31`. That one wraps `pd.read_csv`. CSV content
is a DataFrame, which is data, which is `cache_data`. Same
decorator, same logic.

## 2:30 — The underscore gotcha (5 min)

Look again at line 59:

```python
def annual_sun_path(_loader: EphemerisLoader, year: int):
```

That leading underscore on `_loader` is not a Python convention —
it is a Streamlit convention, and it is *load-bearing.*

`cache_data` decides whether to hit the cache by hashing its
arguments. If any argument can't be hashed, the call explodes with
`UnhashableParamError`. An `EphemerisLoader` instance is not
hashable — there's no sensible way to turn an open file handle
into a number.

The fix: prefix the argument name with an underscore. That tells
Streamlit "don't try to hash this argument — pretend it's not part
of the cache key." The cache key becomes just `year`. That's fine
as long as you're sure `_loader` doesn't actually affect the result
in a way you'd miss. Here it doesn't, because there's only one
loader per process.

This is the single most common "why is my cache broken?" question.
**Underscore prefix = skip this arg when hashing.** Write that on
the board.

## 2:35 — Demo: cold vs warm cache timing (10 min)

Let's prove caching does something. Scratch file:

> **Action:** create `cache_demo.py` in the repo root:
>
> ```python
> import time
> from pathlib import Path
> import pandas as pd
> import streamlit as st
>
> PATH = Path("portwatch/data/chokepoint_transits.csv")
>
> @st.cache_data
> def load():
>     time.sleep(2)  # pretend it's a 2-second load
>     return pd.read_csv(PATH, parse_dates=["date"])
>
> t0 = time.perf_counter()
> df = load()
> st.write(f"Load took {time.perf_counter() - t0:.3f}s — {len(df)} rows")
>
> n = st.slider("Tail rows", 1, 100, 10)
> st.dataframe(df.tail(n))
> ```
>
> Run it: `streamlit run cache_demo.py`. First load: ~2 seconds.
> Move the slider. Subsequent reruns: "0.000s." Cache hit. Only
> `st.slider` and `st.dataframe` actually run.

That's the whole value proposition of `@st.cache_data` in one
screen. One decorator. Expensive function runs once per unique
set of arguments.

> **Optional extra demo:** remove the `@st.cache_data` line. Save.
> Now every slider move takes 2 full seconds. Put it back. Instant
> again.

## 2:45 — When the data doesn't fit in memory at all (10 min)

`cache_data` helps when your data fits in memory after the first
load. What if it doesn't?

Short version: Streamlit is not a big-data tool. If your data is
genuinely huge — hundreds of millions of rows, multi-gigabyte
files — you don't load it into the app. You put it behind a query
engine and the Streamlit app issues queries.

Three practical options:

- **Parquet + DuckDB.** Point DuckDB at a Parquet file or
  directory. Columnar, fast, handles larger-than-memory data for
  many query shapes. This is the best "small team, lots of data"
  answer in 2026. Wrap the DuckDB connection in
  `@st.cache_resource`, wrap the individual query results in
  `@st.cache_data`.
- **A real database.** Postgres, ClickHouse, BigQuery, Snowflake.
  Same split: connection in `cache_resource`, query results in
  `cache_data`.
- **Sharded files on disk** — which is what `solar/` actually
  does.

Go look at `solar/data/` again in the terminal. One binary file
per month. The loader (`EphemerisLoader`) only reads the month it
needs, and its internal cache is hard-capped at 12 months (LRU
evicts the oldest). Result: the app boots instantly, touches
128 KB for the default view, and never grows past ~1.5 MB of
in-memory chunks even though the full dataset on disk is 46 MB
of hourly positions across 31 years.

The loader sits behind `@st.cache_resource` — one instance per
process. Its internal per-month cache is a plain Python dict
inside the loader; Streamlit knows nothing about it. That's the
pattern:

> **Streamlit caches the entry point. Your code does its own
> smart paging underneath.**

And for the derived view — `annual_sun_path`, which is 365
samples per year — we use `@st.cache_data(_loader, year)`. First
time you pick "Annual sun path" for 2024, we pay the computation
cost. After that, it's instant, and you can flip overlays back
and forth as much as you want.

> **Demo:** in the solar app, switch to "Annual sun path." First
> render takes a beat. Switch to "Day / night", then back to
> "Annual sun path." Second render is instant. Change the date
> within the same year — still instant, because the path doesn't
> depend on the specific date, only on the year. Change to
> `2025-06-21`. Now it recomputes for 2025, then caches that too.

One `cache_resource`, one `cache_data`, one underscore, and 30
years of astronomical data loads in under a second on a laptop.

> **Question:** "For a 10 GB sales log with a year of daily
> orders, what do you reach for?"
>
> Expected answer: Parquet + DuckDB. Not `cache_data` on
> `pd.read_csv`.

## 2:55 — Tying portwatch and solar together (5 min)

Look at what both apps share.

Both have a cached data entry point — `load_data` in portwatch,
`get_loader` + `annual_sun_path` in solar. Portwatch's is
`cache_data` because the entry returns a DataFrame; solar's
entry is `cache_resource` because the entry is a loader object.
Different decorators, same idea: **cache where the expensive
thing happens, as close to the data as possible.**

Both have a "filter once" step before views render. Portwatch
makes it explicit with `fdf`. Solar's version is quieter — the
`entry = loader.get(dt)` call at line 100 is the same pattern,
just downsampling to a single instant.

Both have a tail `st.expander` for raw data. Both use Plotly as
the plotting layer. Both use `st.columns` to lay out their
header controls.

If you internalise these shared moves — **cached load →
upstream filter → layered Plotly views → reactive widgets** —
you have a template for 90% of the dashboards you'll build with
Streamlit.

## 3:00 — Wrap-up (5 min)

Three hours, two apps, three big ideas.

1. **The script reruns on every interaction.** That's the entire
   mental model. Everything else falls out.
2. **Context filters are just reactive values.** Global filters
   come from the sidebar. Cross-chart filters come from
   `on_select="rerun"` and the event object returned by
   `st.plotly_chart`. Both feed the same downstream pipeline.
3. **Cache data with `@st.cache_data`, cache objects with
   `@st.cache_resource`.** Underscore-prefix unhashable args.
   Filter upstream once. For genuinely big data, use a query
   engine and cache the results.

Homework, if you want it:

- Take portwatch, drop the `load_data` function, and replace it
  with a DuckDB query against a Parquet version of the CSV.
  Measure the difference.
- Remove the hard-coded Hormuz filter in `load_data` and add a
  chokepoint picker above the date range in the sidebar.
- In solar, add a click-on-body interaction: click the Sun
  marker, show a panel with its zodiac sign, subpoint
  coordinates, and rise/set times for a configurable location.

Any one of those will teach you more than another three hours
of me talking. Go build.

---

## Appendix A — Code pointers for the class

Every teaching moment in the two demo files is marked with a
`# ─── DEMO [Hour N]: … ───` comment — scroll to those during the live
session and speak from them. Current line numbers:

| Concept | File | Line |
|---|---|---|
| `@st.cache_data` on data load | portwatch/app.py | 35 |
| CSV-once / Parquet-forever trick | portwatch/app.py | 52 |
| `set_page_config` must be first | portwatch/app.py | 82 |
| Global filters in the sidebar | portwatch/app.py | 96 |
| `date_input` tuple quirk | portwatch/app.py | 110 |
| **Filter once, upstream of every view** | portwatch/app.py | 130 |
| `st.stop()` on empty filter | portwatch/app.py | 145 |
| `st.columns` + `st.metric` | portwatch/app.py | 171 |
| `st.plotly_chart(on_select="rerun")` | portwatch/app.py | 211 |
| Reading the click-event object | portwatch/app.py | 244 |
| Drill-down branch | portwatch/app.py | 261 |
| `@st.cache_resource` (singleton) | solar/app.py | 105 |
| `@st.cache_data` + underscore-arg gotcha | solar/app.py | 123 |
| Lazy-load moment (`loader.get(dt)`) | solar/app.py | 315 |

Line numbers drift as the files evolve — `grep "DEMO \[Hour"` in either
file gives you the current list.

## Appendix B — Questions students always ask

**"Can I run this without a browser?"** Yes: `streamlit run app.py
--server.headless true`.

**"Does it work offline?"** Fully. No cloud required.

**"How do I deploy it?"** Streamlit Community Cloud for small
public apps. For internal deployment it's a normal Python process
behind a reverse proxy — Docker image, run `streamlit run app.py`,
expose port 8501.

**"Can I use it for login-gated internal tools?"** Yes, but auth is
your problem — put it behind an OIDC proxy or use a third-party
add-on. Streamlit itself does not ship auth.

**"What about multi-page apps?"** Create a `pages/` directory with
one `.py` per page. Streamlit auto-discovers them and adds a nav
to the sidebar.

**"Why did my cache 'not work'?"** Ninety percent of the time it's
the underscore-arg thing, or the function depends on a module-level
global that changed, or you're mutating the cached DataFrame —
always `.copy()` before you mutate.

**"Can I update just one chart without rerunning the whole script?"**
Partially, with `st.fragment` on newer Streamlit versions. But
start by accepting the rerun model; it's almost always fast enough
once you cache correctly.
