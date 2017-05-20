

```python

```


```python
from bokeh.plotting import figure, output_notebook, show
output_notebook()
# prepare some data
x = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
y0 = [i**2 for i in x]
y1 = [10**i for i in x]
y2 = [10**(i**2) for i in x]

# output to static HTML file
# output_file("log_lines.html")

# create a new plot
p = figure(
   tools="pan,box_zoom,reset,save",
   y_axis_type="log", y_range=[0.001, 10**11], title="log axis example",
   x_axis_label='sections', y_axis_label='particles'
)

# add some renderers
p.line(x, x, legend="y=x")
p.circle(x, x, legend="y=x", fill_color="white", size=8)
p.line(x, y0, legend="y=x^2", line_width=3)
p.line(x, y1, legend="y=10^x", line_color="red")
p.circle(x, y1, legend="y=10^x", fill_color="red", line_color="red", size=6)
p.line(x, y2, legend="y=10^x^2", line_color="orange", line_dash="4 4")

# show the results
show(p)
```



    <div class="bk-root">
        <a href="http://bokeh.pydata.org" target="_blank" class="bk-logo bk-logo-small bk-logo-notebook"></a>
        <span id="384610e6-a115-41c6-a88b-226b14318059">Loading BokehJS ...</span>
    </div>







    <div class="bk-root">
        <div class="plotdiv" id="c0a2dec1-e328-4d44-8a5f-41a94b08fe62"></div>
    </div>
<script type="text/javascript">
  
  (function(global) {
    function now() {
      return new Date();
    }
  
    var force = "";
  
    if (typeof (window._bokeh_onload_callbacks) === "undefined" || force !== "") {
      window._bokeh_onload_callbacks = [];
      window._bokeh_is_loading = undefined;
    }
  
  
    
    if (typeof (window._bokeh_timeout) === "undefined" || force !== "") {
      window._bokeh_timeout = Date.now() + 0;
      window._bokeh_failed_load = false;
    }
  
    var NB_LOAD_WARNING = {'data': {'text/html':
       "<div style='background-color: #fdd'>\n"+
       "<p>\n"+
       "BokehJS does not appear to have successfully loaded. If loading BokehJS from CDN, this \n"+
       "may be due to a slow or bad network connection. Possible fixes:\n"+
       "</p>\n"+
       "<ul>\n"+
       "<li>re-rerun `output_notebook()` to attempt to load from CDN again, or</li>\n"+
       "<li>use INLINE resources instead, as so:</li>\n"+
       "</ul>\n"+
       "<code>\n"+
       "from bokeh.resources import INLINE\n"+
       "output_notebook(resources=INLINE)\n"+
       "</code>\n"+
       "</div>"}};
  
    function display_loaded() {
      if (window.Bokeh !== undefined) {
        Bokeh.$("#c0a2dec1-e328-4d44-8a5f-41a94b08fe62").text("BokehJS successfully loaded.");
      } else if (Date.now() < window._bokeh_timeout) {
        setTimeout(display_loaded, 100)
      }
    }
  
    function run_callbacks() {
      window._bokeh_onload_callbacks.forEach(function(callback) { callback() });
      delete window._bokeh_onload_callbacks
      console.info("Bokeh: all callbacks have finished");
    }
  
    function load_libs(js_urls, callback) {
      window._bokeh_onload_callbacks.push(callback);
      if (window._bokeh_is_loading > 0) {
        console.log("Bokeh: BokehJS is being loaded, scheduling callback at", now());
        return null;
      }
      if (js_urls == null || js_urls.length === 0) {
        run_callbacks();
        return null;
      }
      console.log("Bokeh: BokehJS not loaded, scheduling load and callback at", now());
      window._bokeh_is_loading = js_urls.length;
      for (var i = 0; i < js_urls.length; i++) {
        var url = js_urls[i];
        var s = document.createElement('script');
        s.src = url;
        s.async = false;
        s.onreadystatechange = s.onload = function() {
          window._bokeh_is_loading--;
          if (window._bokeh_is_loading === 0) {
            console.log("Bokeh: all BokehJS libraries loaded");
            run_callbacks()
          }
        };
        s.onerror = function() {
          console.warn("failed to load library " + url);
        };
        console.log("Bokeh: injecting script tag for BokehJS library: ", url);
        document.getElementsByTagName("head")[0].appendChild(s);
      }
    };var element = document.getElementById("c0a2dec1-e328-4d44-8a5f-41a94b08fe62");
    if (element == null) {
      console.log("Bokeh: ERROR: autoload.js configured with elementid 'c0a2dec1-e328-4d44-8a5f-41a94b08fe62' but no matching script tag was found. ")
      return false;
    }
  
    var js_urls = [];
  
    var inline_js = [
      function(Bokeh) {
        Bokeh.$(function() {
            var docs_json = {"9b9146c1-4b12-4ca4-9ffd-af7ea9b2c252":{"roots":{"references":[{"attributes":{"num_minor_ticks":10},"id":"bb3ed8bb-890f-48ae-b5de-d0a9ae9efe94","type":"LogTicker"},{"attributes":{"overlay":{"id":"5e07a718-f226-483e-8882-144127bacd73","type":"BoxAnnotation"},"plot":{"id":"82416a29-ed89-4cd9-9430-f36349599521","subtype":"Figure","type":"Plot"}},"id":"3202de13-19d1-4460-b221-0b3a1aca7f0a","type":"BoxZoomTool"},{"attributes":{"plot":{"id":"82416a29-ed89-4cd9-9430-f36349599521","subtype":"Figure","type":"Plot"}},"id":"97baea0b-1295-4ef8-9593-ebb50716a83d","type":"PanTool"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b4"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":8},"x":{"field":"x"},"y":{"field":"y"}},"id":"0e30fc4f-d531-4c29-84db-bbcca9b73c52","type":"Circle"},{"attributes":{"legends":[["y=x",[{"id":"17fc7f0c-13e2-4192-9635-ae5796a5d812","type":"GlyphRenderer"},{"id":"0d9035ac-cbac-457f-9c3d-a6a01c36e497","type":"GlyphRenderer"}]],["y=x^2",[{"id":"68a6c37b-72c0-486e-9b7c-8ca647e7d00e","type":"GlyphRenderer"}]],["y=10^x",[{"id":"68c87c74-240c-4ae3-a2c0-cfb496bf6d6e","type":"GlyphRenderer"},{"id":"65a499a3-3055-4408-b5a9-392acd27cd95","type":"GlyphRenderer"}]],["y=10^x^2",[{"id":"37595df3-7fed-405d-8021-909f7cb7953d","type":"GlyphRenderer"}]]],"plot":{"id":"82416a29-ed89-4cd9-9430-f36349599521","subtype":"Figure","type":"Plot"}},"id":"f0ef9c92-2850-4a68-b444-7c4ca08d791b","type":"Legend"},{"attributes":{"callback":null,"column_names":["y","x"],"data":{"x":[0.1,0.5,1.0,1.5,2.0,2.5,3.0],"y":[1.2589254117941673,3.1622776601683795,10.0,31.622776601683793,100.0,316.22776601683796,1000.0]}},"id":"90458e51-cab2-4826-9bf8-4be093512041","type":"ColumnDataSource"},{"attributes":{},"id":"00ecf594-ebe3-487b-8a56-1f1f35c9e93f","type":"ToolEvents"},{"attributes":{"line_color":{"value":"red"},"x":{"field":"x"},"y":{"field":"y"}},"id":"2635f0fe-cfe2-48a2-ab38-211de6c67972","type":"Line"},{"attributes":{"callback":null},"id":"1651a436-f12a-4131-b130-6ac9c0bd1e2a","type":"DataRange1d"},{"attributes":{"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"line_width":{"value":3},"x":{"field":"x"},"y":{"field":"y"}},"id":"967aed21-a346-4fdd-b6d7-bee4be60cd2c","type":"Line"},{"attributes":{"fill_color":{"value":"white"},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":8},"x":{"field":"x"},"y":{"field":"y"}},"id":"e0d3c893-920c-48e3-8cc5-34e783ce266a","type":"Circle"},{"attributes":{"active_drag":"auto","active_scroll":"auto","active_tap":"auto","tools":[{"id":"97baea0b-1295-4ef8-9593-ebb50716a83d","type":"PanTool"},{"id":"3202de13-19d1-4460-b221-0b3a1aca7f0a","type":"BoxZoomTool"},{"id":"f45b5afa-1b86-465a-89cd-12de58dd3b4e","type":"ResetTool"},{"id":"2ab6a665-cc0f-45fe-aaca-9784aaecd6a7","type":"SaveTool"}]},"id":"4fa27d4a-7f28-4d91-8036-4da7ce593ff0","type":"Toolbar"},{"attributes":{"plot":{"id":"82416a29-ed89-4cd9-9430-f36349599521","subtype":"Figure","type":"Plot"}},"id":"f45b5afa-1b86-465a-89cd-12de58dd3b4e","type":"ResetTool"},{"attributes":{"line_color":{"value":"#1f77b4"},"line_width":{"value":3},"x":{"field":"x"},"y":{"field":"y"}},"id":"927ef702-1abc-401c-8a10-8590471e65f2","type":"Line"},{"attributes":{"callback":null,"column_names":["y","x"],"data":{"x":[0.1,0.5,1.0,1.5,2.0,2.5,3.0],"y":[1.023292992280754,1.7782794100389228,10.0,177.82794100389228,10000.0,1778279.410038923,1000000000.0]}},"id":"48bfdcfe-142a-4e38-8718-a78ea75c80c7","type":"ColumnDataSource"},{"attributes":{"data_source":{"id":"a1c08235-e920-4782-a6b6-ee7e7e67cd93","type":"ColumnDataSource"},"glyph":{"id":"ed3df505-b4c8-4b2f-bc8c-51ed873d9c3d","type":"Line"},"hover_glyph":null,"nonselection_glyph":{"id":"5a5837c5-4cd8-4030-b776-4714eb800dab","type":"Line"},"selection_glyph":null},"id":"17fc7f0c-13e2-4192-9635-ae5796a5d812","type":"GlyphRenderer"},{"attributes":{"fill_color":{"value":"red"},"line_color":{"value":"red"},"size":{"units":"screen","value":6},"x":{"field":"x"},"y":{"field":"y"}},"id":"5acca2f9-dfec-41f0-81d2-558443fad1af","type":"Circle"},{"attributes":{"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"x":{"field":"x"},"y":{"field":"y"}},"id":"8c82412b-da96-46a5-be94-abb8595eefd8","type":"Line"},{"attributes":{"data_source":{"id":"48bfdcfe-142a-4e38-8718-a78ea75c80c7","type":"ColumnDataSource"},"glyph":{"id":"acfc6d36-aeea-4192-85bf-ce77676834be","type":"Line"},"hover_glyph":null,"nonselection_glyph":{"id":"7c5892b8-a0f4-4348-b83d-75b5cd843ea9","type":"Line"},"selection_glyph":null},"id":"37595df3-7fed-405d-8021-909f7cb7953d","type":"GlyphRenderer"},{"attributes":{"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"x":{"field":"x"},"y":{"field":"y"}},"id":"5a5837c5-4cd8-4030-b776-4714eb800dab","type":"Line"},{"attributes":{"line_color":{"value":"#1f77b4"},"x":{"field":"x"},"y":{"field":"y"}},"id":"ed3df505-b4c8-4b2f-bc8c-51ed873d9c3d","type":"Line"},{"attributes":{"data_source":{"id":"e264ed07-92e2-4241-8a66-52669228880b","type":"ColumnDataSource"},"glyph":{"id":"5acca2f9-dfec-41f0-81d2-558443fad1af","type":"Circle"},"hover_glyph":null,"nonselection_glyph":{"id":"355cbdb6-977d-4f4b-82f1-d71733c88677","type":"Circle"},"selection_glyph":null},"id":"65a499a3-3055-4408-b5a9-392acd27cd95","type":"GlyphRenderer"},{"attributes":{"dimension":1,"plot":{"id":"82416a29-ed89-4cd9-9430-f36349599521","subtype":"Figure","type":"Plot"},"ticker":{"id":"bb3ed8bb-890f-48ae-b5de-d0a9ae9efe94","type":"LogTicker"}},"id":"f70d2cb6-25e1-45be-ad8d-f1fbe586af16","type":"Grid"},{"attributes":{},"id":"fbf7ee41-40cf-4fa7-a9cb-42c9cac25710","type":"BasicTicker"},{"attributes":{"callback":null,"column_names":["y","x"],"data":{"x":[0.1,0.5,1.0,1.5,2.0,2.5,3.0],"y":[0.010000000000000002,0.25,1.0,2.25,4.0,6.25,9.0]}},"id":"4e1208c7-b688-4b12-a84f-825065fee5cb","type":"ColumnDataSource"},{"attributes":{"data_source":{"id":"4e1208c7-b688-4b12-a84f-825065fee5cb","type":"ColumnDataSource"},"glyph":{"id":"927ef702-1abc-401c-8a10-8590471e65f2","type":"Line"},"hover_glyph":null,"nonselection_glyph":{"id":"967aed21-a346-4fdd-b6d7-bee4be60cd2c","type":"Line"},"selection_glyph":null},"id":"68a6c37b-72c0-486e-9b7c-8ca647e7d00e","type":"GlyphRenderer"},{"attributes":{"plot":{"id":"82416a29-ed89-4cd9-9430-f36349599521","subtype":"Figure","type":"Plot"}},"id":"2ab6a665-cc0f-45fe-aaca-9784aaecd6a7","type":"SaveTool"},{"attributes":{"ticker":null},"id":"a56f0b07-45ad-4efe-aa73-9f07e95b1ab1","type":"LogTickFormatter"},{"attributes":{"bottom_units":"screen","fill_alpha":{"value":0.5},"fill_color":{"value":"lightgrey"},"left_units":"screen","level":"overlay","line_alpha":{"value":1.0},"line_color":{"value":"black"},"line_dash":[4,4],"line_width":{"value":2},"plot":null,"render_mode":"css","right_units":"screen","top_units":"screen"},"id":"5e07a718-f226-483e-8882-144127bacd73","type":"BoxAnnotation"},{"attributes":{"data_source":{"id":"90458e51-cab2-4826-9bf8-4be093512041","type":"ColumnDataSource"},"glyph":{"id":"2635f0fe-cfe2-48a2-ab38-211de6c67972","type":"Line"},"hover_glyph":null,"nonselection_glyph":{"id":"8c82412b-da96-46a5-be94-abb8595eefd8","type":"Line"},"selection_glyph":null},"id":"68c87c74-240c-4ae3-a2c0-cfb496bf6d6e","type":"GlyphRenderer"},{"attributes":{"callback":null,"column_names":["x","y"],"data":{"x":[0.1,0.5,1.0,1.5,2.0,2.5,3.0],"y":[1.2589254117941673,3.1622776601683795,10.0,31.622776601683793,100.0,316.22776601683796,1000.0]}},"id":"e264ed07-92e2-4241-8a66-52669228880b","type":"ColumnDataSource"},{"attributes":{"line_color":{"value":"orange"},"line_dash":[4,4],"x":{"field":"x"},"y":{"field":"y"}},"id":"acfc6d36-aeea-4192-85bf-ce77676834be","type":"Line"},{"attributes":{"plot":{"id":"82416a29-ed89-4cd9-9430-f36349599521","subtype":"Figure","type":"Plot"},"ticker":{"id":"fbf7ee41-40cf-4fa7-a9cb-42c9cac25710","type":"BasicTicker"}},"id":"f665753b-b542-4838-8cbd-22bb4a55ceb1","type":"Grid"},{"attributes":{"below":[{"id":"7c7a573b-dd0c-4eb1-8eb9-a70950850de2","type":"LinearAxis"}],"left":[{"id":"2fa423d3-7603-4b38-911e-595b000a1119","type":"LogAxis"}],"renderers":[{"id":"7c7a573b-dd0c-4eb1-8eb9-a70950850de2","type":"LinearAxis"},{"id":"f665753b-b542-4838-8cbd-22bb4a55ceb1","type":"Grid"},{"id":"2fa423d3-7603-4b38-911e-595b000a1119","type":"LogAxis"},{"id":"f70d2cb6-25e1-45be-ad8d-f1fbe586af16","type":"Grid"},{"id":"5e07a718-f226-483e-8882-144127bacd73","type":"BoxAnnotation"},{"id":"f0ef9c92-2850-4a68-b444-7c4ca08d791b","type":"Legend"},{"id":"17fc7f0c-13e2-4192-9635-ae5796a5d812","type":"GlyphRenderer"},{"id":"0d9035ac-cbac-457f-9c3d-a6a01c36e497","type":"GlyphRenderer"},{"id":"68a6c37b-72c0-486e-9b7c-8ca647e7d00e","type":"GlyphRenderer"},{"id":"68c87c74-240c-4ae3-a2c0-cfb496bf6d6e","type":"GlyphRenderer"},{"id":"65a499a3-3055-4408-b5a9-392acd27cd95","type":"GlyphRenderer"},{"id":"37595df3-7fed-405d-8021-909f7cb7953d","type":"GlyphRenderer"}],"title":{"id":"4f7a6d6e-2762-411b-922c-dd695ba8c173","type":"Title"},"tool_events":{"id":"00ecf594-ebe3-487b-8a56-1f1f35c9e93f","type":"ToolEvents"},"toolbar":{"id":"4fa27d4a-7f28-4d91-8036-4da7ce593ff0","type":"Toolbar"},"x_range":{"id":"1651a436-f12a-4131-b130-6ac9c0bd1e2a","type":"DataRange1d"},"y_mapper_type":"log","y_range":{"id":"8729b713-6eab-4915-802f-3cd30c358971","type":"Range1d"}},"id":"82416a29-ed89-4cd9-9430-f36349599521","subtype":"Figure","type":"Plot"},{"attributes":{"axis_label":"sections","formatter":{"id":"a5100d49-157a-4bf8-b05c-7fd7ddc101fd","type":"BasicTickFormatter"},"plot":{"id":"82416a29-ed89-4cd9-9430-f36349599521","subtype":"Figure","type":"Plot"},"ticker":{"id":"fbf7ee41-40cf-4fa7-a9cb-42c9cac25710","type":"BasicTicker"}},"id":"7c7a573b-dd0c-4eb1-8eb9-a70950850de2","type":"LinearAxis"},{"attributes":{"callback":null,"column_names":["y","x"],"data":{"x":[0.1,0.5,1.0,1.5,2.0,2.5,3.0],"y":[0.1,0.5,1.0,1.5,2.0,2.5,3.0]}},"id":"02d4ecc3-0d45-4b72-a217-8c1b9f522727","type":"ColumnDataSource"},{"attributes":{"axis_label":"particles","formatter":{"id":"a56f0b07-45ad-4efe-aa73-9f07e95b1ab1","type":"LogTickFormatter"},"plot":{"id":"82416a29-ed89-4cd9-9430-f36349599521","subtype":"Figure","type":"Plot"},"ticker":{"id":"bb3ed8bb-890f-48ae-b5de-d0a9ae9efe94","type":"LogTicker"}},"id":"2fa423d3-7603-4b38-911e-595b000a1119","type":"LogAxis"},{"attributes":{},"id":"a5100d49-157a-4bf8-b05c-7fd7ddc101fd","type":"BasicTickFormatter"},{"attributes":{"callback":null,"column_names":["y","x"],"data":{"x":[0.1,0.5,1.0,1.5,2.0,2.5,3.0],"y":[0.1,0.5,1.0,1.5,2.0,2.5,3.0]}},"id":"a1c08235-e920-4782-a6b6-ee7e7e67cd93","type":"ColumnDataSource"},{"attributes":{"callback":null,"end":100000000000,"start":0.001},"id":"8729b713-6eab-4915-802f-3cd30c358971","type":"Range1d"},{"attributes":{"plot":null,"text":"log axis example"},"id":"4f7a6d6e-2762-411b-922c-dd695ba8c173","type":"Title"},{"attributes":{"data_source":{"id":"02d4ecc3-0d45-4b72-a217-8c1b9f522727","type":"ColumnDataSource"},"glyph":{"id":"e0d3c893-920c-48e3-8cc5-34e783ce266a","type":"Circle"},"hover_glyph":null,"nonselection_glyph":{"id":"0e30fc4f-d531-4c29-84db-bbcca9b73c52","type":"Circle"},"selection_glyph":null},"id":"0d9035ac-cbac-457f-9c3d-a6a01c36e497","type":"GlyphRenderer"},{"attributes":{"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"line_dash":[4,4],"x":{"field":"x"},"y":{"field":"y"}},"id":"7c5892b8-a0f4-4348-b83d-75b5cd843ea9","type":"Line"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b4"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":6},"x":{"field":"x"},"y":{"field":"y"}},"id":"355cbdb6-977d-4f4b-82f1-d71733c88677","type":"Circle"}],"root_ids":["82416a29-ed89-4cd9-9430-f36349599521"]},"title":"Bokeh Application","version":"0.12.2"}};
            var render_items = [{"docid":"9b9146c1-4b12-4ca4-9ffd-af7ea9b2c252","elementid":"c0a2dec1-e328-4d44-8a5f-41a94b08fe62","modelid":"82416a29-ed89-4cd9-9430-f36349599521"}];
            
            Bokeh.embed.embed_items(docs_json, render_items);
        });
      },
      function(Bokeh) {
      }
    ];
  
    function run_inline_js() {
      
      if ((window.Bokeh !== undefined) || (force === "1")) {
        for (var i = 0; i < inline_js.length; i++) {
          inline_js[i](window.Bokeh);
        }if (force === "1") {
          display_loaded();
        }} else if (Date.now() < window._bokeh_timeout) {
        setTimeout(run_inline_js, 100);
      } else if (!window._bokeh_failed_load) {
        console.log("Bokeh: BokehJS failed to load within specified timeout.");
        window._bokeh_failed_load = true;
      } else if (!force) {
        var cell = $("#c0a2dec1-e328-4d44-8a5f-41a94b08fe62").parents('.cell').data().cell;
        cell.output_area.append_execute_result(NB_LOAD_WARNING)
      }
  
    }
  
    if (window._bokeh_is_loading === 0) {
      console.log("Bokeh: BokehJS loaded, going straight to plotting");
      run_inline_js();
    } else {
      load_libs(js_urls, function() {
        console.log("Bokeh: BokehJS plotting callback run at", now());
        run_inline_js();
      });
    }
  }(this));
</script>



```python

```
