(function (d3$1, topojson) {
  'use strict';

  topojson = topojson && Object.prototype.hasOwnProperty.call(topojson, 'default') ? topojson['default'] : topojson;

  const colorData = (min, max, num) => {
    const scale = d3$1.scaleSequential(d3$1.interpolateYlGnBu).domain([min, max]);
    return scale(num);
  };
  const projection = d3$1.geoNaturalEarth1();
  const path = d3$1.geoPath(projection);
  const graticule = d3$1.geoGraticule();

  const tooltip = d3
    .select('body')
    .append('div')
    .attr('class', 'd3-tooltip')
    .style('position', 'absolute')
    .style('z-index', '10')
    .style('visibility', 'hidden')
    .style('padding', '10px')
    .style('background', 'rgba(0,0,0,0.6)')
    .style('border-radius', '5px')
    .style('color', 'white');

  const map = (selection, {data}) => {
    fetch('https://raw.githubusercontent.com/stanleyshen2003/Data-Visualization/main/FinalProject/data.csv')
      .then((response) => response.text())
      .then((csvString) => {
        let internet = d3$1.csvParse(csvString);
        function valuePerCountry(country) {
          let value = 'No Data';
          internet.forEach((c) => {
            if (c.country == country && c.year == M_year) {
              value = c[M_feature];
            }
          });
          return value;
        }
        function getColor(num) {
          if (isNaN(num) || num < M_Legend_start || num > M_Legend_end) {
            return 'lightgray';
          }
          let color = colorData(M_Value_min, M_Value_max, num);
          return color;
        }
        selection.attr('width',Svg_width-10)
        const mapGroup = selection
          .append('g')
          .attr('transform', `translate(40, 35) scale(${0.9})`);

        mapGroup
          .selectAll('path.graticule')
          .data([null])
          .join('path')
          .attr('class', 'graticule')
          .attr('d', path(graticule()))
          .attr('fill', 'none')
          .attr('stroke', '#BBB')
          .attr('stroke-width', 0.5);

        mapGroup
          .selectAll('path.outline')
          .data([null])
          .join('path')
          .attr('class', 'outline')
          .attr('d', path(graticule.outline()))
          .attr('fill', '#d6ebff')
          .attr('stroke', 'black')
          .attr('stroke-width', 2);

        mapGroup
          .selectAll('path.country')
          .data(data.features)
          .join('path')
          .attr('d', path)
          .attr('class', 'country')
          .attr('stroke', 'black')
          .attr('fill', (d) => getColor(valuePerCountry(d.properties.name)))
          .attr('stroke-width', 0.5)
          .style('cursor', 'default')
          .on('mouseover', function (e, d) {
            let count = valuePerCountry(d.properties.name);
            if (count != 'No Data') {
              count = count ;
            }
            tooltip
              .html(`${d.properties.name}: ${count}`)
              .style('visibility', 'visible');
          })
          .on('mousemove', function () {
            tooltip
              .style('top', event.pageY - 10 + 'px')
              .style('left', event.pageX + 10 + 'px');
          })
          .on('mouseout', function () {
            tooltip.style('visibility', 'hidden');
          })
          .on('click', function (e, d) {
             D_country = d.properties.name;
             d3.select('#D_input').property('value', D_country );
             d3.select('input[name="D_Country_Btn"][value="country"]').property('checked', true);
             d3.select('input[name="D_Data_Btn"][value="normalize"]').property('disabled', false);
             tooltip.style('visibility', 'hidden');
             d3.select('#Nav_Btn_D').dispatch('click');
          });

        const zoom = d3
          .zoom()
          .scaleExtent([1, 10])
          .on('zoom', zoomed);

        selection.call(zoom);

        function zoomed(event) {
          const { transform } = event;          
          selection.attr('transform', transform);
        }
           
      });
    
    
  };

  const worldAtlasURL =
    'https://unpkg.com/visionscarto-world-atlas@0.1.0/world/50m.json';

  const viz = (
    container,
    { state, setState}
    
  ) => {
    
    const width = Svg_width;
    const height = Svg_height;

    const svg = d3$1.select(container)
      .selectAll('svg')
      .data([null])
      .join('svg')
      .attr('width', width)
      .attr('height', height);

    const { data } = state;

    if (data && data !== 'LOADING') {
      svg.call(map, { data} );
    }
    if (data === undefined) {
      setState((state) => ({
        ...state,
        data: 'LOADING',
      }) );
      fetch(worldAtlasURL)
        .then((response) => response.json())
        .then((topoJSONData) => {
          const data = topojson.feature(
            topoJSONData,
            'countries'
          );
          setState((state) => ({
            ...state,
            data,
          }));
        });
    }
  };

  const svg_main = d3$1.select('#Svg_Main').node();
  let state = {};

  const Georender = () => {
    
    viz(svg_main, {
      state,
      setState}
    );
  };

  const setState = (next) => {

    state = next(state);
    Georender();
  };

  //Init ========================================================================================
  var M_feature = 'happiness';
  var M_year = 2023;
  var M_Value_min = 0;
  var M_Value_max = 100;
  var M_Legend_start = 0;
  var M_Legend_end = 100;
  const M_width = d3
    .select('#Svg_Tool')
    .node()
    .getBoundingClientRect().width;

  var R_feature = 'happiness';
  var R_year = 2023;
  var R_Min = 1;
  var R_Max = 20;
  var R_Value_max = 100;
  var R_region = 'all';
  
  var R_tooltip = d3
      .select('body')
      .append('div')
      .attr('class', 'tooltip')
      .style('opacity', 0)
      .style('position', 'absolute')
      .style('pointer-events', 'none')
      .style('padding', '10px')
      .style('background', 'rgba(0,0,0,0.6)')
      .style('border-radius', '5px')
      .style('color', 'white');
  

  const C_width = d3
    .select('#Svg_Tool')
    .node()
    .getBoundingClientRect().width;
  var C_Value_min = -1;
  var C_Value_max = 1;
  var C_tooltip = d3
    .select('body')
    .append('div')
    .attr('class', 'tooltip')
    .style('opacity', 0)
    .style('position', 'absolute')
    .style('pointer-events', 'none')
    .style('padding', '10px')
    .style('background', 'rgba(0,0,0,0.6)')
    .style('border-radius', '5px')
    .style('color', 'white');
    

  var D_country = 'year_average';
  var D_country_C = 'year_average';
  var D_flag = 'raw';
  const D_width = d3
    .select('#Svg_Tool')
    .node()
    .getBoundingClientRect().width;
  var D_tooltip = d3
      .select('body')
      .append('div')
      .attr('class', 'tooltip')
      .style('opacity', 0)
      .style('position', 'absolute')
      .style('pointer-events', 'none')
      .style('padding', '10px')
      .style('background', 'rgba(0,0,0,0.6)')
      .style('border-radius', '5px')
      .style('color', 'white')
      .style('line-height', '1.5');

  var Svg_width = d3
    .select('#Svg_Main')
    .node()
    .getBoundingClientRect().width;
  var Svg_height = d3
    .select('#Svg_Main')
    .node()
    .getBoundingClientRect().height;

  var D_keys = [
    'happiness',
    'economy',
    'family',
    'health',
    'freedom',
    'trust',
    'generosity',
  ];

  var D_color = d3
    .scaleOrdinal()
    .domain(D_keys)
    .range(d3.schemeSet2);

  // LoadData====================================================================================
  var dataG;
  var cty = [];
  var cty_ori = [];
  var reg = [];
  d3.csv('https://raw.githubusercontent.com/yihsuan29/data/main/data.csv').then((data) => {
    data = data.map(function (d) {
      if (!cty.includes(d.country)) {
        cty.push(d.country);
      }

      if (!reg.includes(d.region)) {
        reg.push(d.region);
      }
      return {
        country: d.country,
        region: d.region,
        happiness_rank: +d.happiness_rank,
        happiness: +d.happiness,
        economy: +d.economy,
        family: +d.family,
        health: +d.health,
        freedom: +d.freedom,
        trust: +d.trust,
        generosity: +d.generosity,
        year: +d.year,
      };
    });
    dataG = data;
    cty_ori = cty;
    cty = cty.filter(value => value !== "year_average");
    R_Value_max = cty.length;
    M_update();
    Georender( );
  });

  function M_update() {
    var dataY = dataG.filter(function (d) {
      return d.year === M_year;
    });

    M_Value_min = d3.min(dataG, function (d) {
      return d[M_feature];
    });

    M_Legend_start = M_Value_min;

    M_Value_max = d3.max(dataG, function (d) {
      return d[M_feature];
    });

    M_Legend_end = M_Value_max;
    M_Legend_g.selectAll('.tick text').text(function (d) {
      return (
        (d * (M_Value_max - M_Value_min)) / 100 +
        M_Value_min
      ).toFixed(2);
    });

    M_Legend_rect1Label.text(M_Legend_start.toFixed(2));

    M_Legend_rect2Labe1.text(M_Legend_end.toFixed(2));
  }

  //Navigation Bar ==============================================================================
  const buttons = d3.selectAll('.Nav_Btn');
  const svgTools = d3.selectAll('.Tool_Container');
  const svgMain = d3.selectAll('#Svg_Main');

  buttons.on('click', function () {
    buttons.classed('active', false);
    svgTools.classed('active', false);
    d3.select(this).classed('active', true);
    svgTools.style('display', 'none');

    const buttonId = d3.select(this).attr('id');
    const index = buttonId.split('_')[2];
    const svgToolId = `Svg_Tool_${index}`;
    d3.select(`#${svgToolId}`).style('display', 'block');

    if (buttonId == 'Nav_Btn_M') {
      svgMain.selectAll('*').remove();
      Georender();
    } else if (buttonId == 'Nav_Btn_R') {
      svgMain.selectAll('*').remove();
      BarRender();
    } else if (buttonId == 'Nav_Btn_D') {
      svgMain.selectAll('*').remove();
      LineChart();
    } else if (buttonId == 'Nav_Btn_C') {
      svgMain.selectAll('*').remove();
      var Svg = svgMain.append('div');
      RenderHeatmap(dataG, Svg, Svg_width, Svg_height);
    }
  });

  // Tool_M =====================================================================================
  // Btns ==========================================================================
  const M_Btns = d3.selectAll(
    '#M_Feature_Btns input[type="radio"]',
  );

  M_Btns.on('change', function () {
    M_feature = d3.select(this).attr('value');
    M_update();
    Georender();
  });

  // Time sliders ==================================================================
  var M_slider = d3
    .sliderHorizontal()
    .min(2015)
    .max(2023)
    .default(2023)
    .step(1)
    .width(M_width-60)
    .displayValue(false)
    .on('onchange', () => {
      M_year = M_slider.value();
      M_update();
      Georender();
    });

  d3.select('#M_Year_Slider')
    .append('svg')
    .attr('width', M_width-25)
    .attr('height', 45)
    .append('g')
    .attr('transform', 'translate(15,10)')
    .classed('slider', true)
    .call(M_slider);

  // Legend =========================================================================
  const M_Legend_width = M_width - 60;
  const M_Legend_height = 20;

  var M_Legend_svg = d3
    .select('#M_Legend_Svg')
    .attr('width', M_Legend_width + 40)
    .attr('height', M_Legend_height + 100);

  var M_Legend_g = M_Legend_svg.append('g').attr(
    'transform',
    'translate(15, 20)',
  );

  var M_Legend_defs = M_Legend_g.append('defs');

  var M_Legend_gradient = M_Legend_defs.append(
    'linearGradient',
  ).attr('id', 'linear-gradient-M');

  var colorScale = d3
    .scaleLinear()
    .domain([0,20,40,60,80, 100])
    .range([
      d3.interpolateYlGnBu(0),
      d3.interpolateYlGnBu(0.2),
      d3.interpolateYlGnBu(0.4),
      d3.interpolateYlGnBu(0.6),
      d3.interpolateYlGnBu(0.8),
      d3.interpolateYlGnBu(1),
    ]);

  var M_Legend_stops = [
    { offset: 0, color: colorScale(0), value: 0 },
    { offset: 0.2, color: colorScale(20), value: 20 },
    { offset: 0.4, color: colorScale(40), value: 40 },
    { offset: 0.6, color: colorScale(60), value: 60 },
    { offset: 0.8, color: colorScale(80), value: 80 },
    { offset: 1, color: colorScale(100), value: 100 },
  ];

  M_Legend_gradient.selectAll('stop')
    .data(M_Legend_stops)
    .enter()
    .append('stop')
    .attr('offset', function (d) {
      return 100 * d.offset + '%';
    })
    .attr('stop-color', function (d) {
      return d.color;
    });

  M_Legend_g.append('rect')
    .attr('width', M_Legend_width)
    .attr('height', M_Legend_height)
    .style('fill', 'url(#linear-gradient-M)');

  var M_Legend_Xscale = d3
    .scaleLinear()
    .domain([0, 100])
    .range([0, M_Legend_width]);

  var M_Legend_axis = d3
    .axisBottom(M_Legend_Xscale)
    .ticks(5)
    .tickSize(5)
    .tickFormat(function customTickFormat(d) {
      return (
        (d * (M_Value_max - M_Value_min)) / 100 + M_Value_min
      );
    });

  M_Legend_g.append('g')
    .attr('transform', 'translate(0,' + M_Legend_height + ')')
    .call(M_Legend_axis);

  var M_Legend_ori_start = 0;
  var M_Legend_ori_end = 100;
  var M_Legend_rect1 = M_Legend_g.append('rect')
    .attr('x', M_Legend_Xscale(0) - 10)
    .attr('y', -1)
    .attr(
      'width',
      M_Legend_Xscale(M_Legend_ori_start) -
        M_Legend_Xscale(0) +
        10,
    )
    .attr('height', M_Legend_height + 2)
    .attr('fill', 'lightgrey')
    .classed('left-rect', true);

  var M_Legend_rect1Label = M_Legend_g.append('text')
    .attr('x', M_Legend_Xscale(M_Legend_ori_start) - 10)
    .attr('y', -5)
    .classed('legend-label-text', true)
    .text(
      (
        (M_Legend_ori_start * (M_Value_max - M_Value_min)) /
          100 +
        M_Value_min
      ).toFixed(2),
    );

  var M_Legend_rect2 = M_Legend_g.append('rect')
    .attr('x', M_Legend_Xscale(M_Legend_ori_end))
    .attr('y', -1)
    .attr(
      'width',
      M_Legend_Xscale(100) -
        M_Legend_Xscale(M_Legend_ori_end) +
        10,
    )
    .attr('height', M_Legend_height + 2)
    .attr('fill', 'lightgrey')
    .classed('right-rect', true);

  var M_Legend_rect2Labe1 = M_Legend_g.append('text')
    .attr('x', M_Legend_Xscale(M_Legend_ori_end) - 10)
    .attr('y', -5)
    .classed('legend-label-text', true)
    .text(
      (
        (M_Legend_ori_end * (M_Value_max - M_Value_min)) /
          100 +
        M_Value_min
      ).toFixed(2),
    );

  var M_Legend_drag1 = d3
    .drag()
    .on('drag', dragging1)
    .on('end', dragEnded1);

  var M_Legend_drag2 = d3
    .drag()
    .on('drag', dragging2)
    .on('end', dragEnded2);

  function dragging1(event) {
    var dragDistance = M_Legend_Xscale.invert(event.dx);
    var new_Legend_ori_start =
      M_Legend_ori_start + dragDistance;
    if (
      new_Legend_ori_start > 0 &&
      new_Legend_ori_start < M_Legend_ori_end
    ) {
      M_Legend_ori_start = new_Legend_ori_start;
      M_Legend_start = (
        (M_Legend_ori_start * (M_Value_max - M_Value_min)) /
          100 +
        M_Value_min
      ).toFixed(2);

      M_Legend_rect1.attr(
        'width',
        M_Legend_Xscale(M_Legend_ori_start) -
          M_Legend_Xscale(0) +
          10,
      );
      M_Legend_rect1Label.attr(
        'x',
        M_Legend_Xscale(M_Legend_ori_start),
      ).text(M_Legend_start);
    }
  }

  function dragEnded1() {
    Georender();
  }
  function dragging2(event) {
    var dragDistance = M_Legend_Xscale.invert(event.dx);
    var new_Legend_ori_end = M_Legend_ori_end + dragDistance;
    if (
      new_Legend_ori_end <= 100 &&
      new_Legend_ori_end > M_Legend_ori_start
    ) {
      M_Legend_ori_end = new_Legend_ori_end;
      M_Legend_end = (
        (M_Legend_ori_end * (M_Value_max - M_Value_min)) /
          100 +
        M_Value_min
      ).toFixed(2);

      M_Legend_rect2.attr(
        'x',
        M_Legend_Xscale(M_Legend_ori_end),
      );
      M_Legend_rect2.attr(
        'width',
        M_Legend_Xscale(100) -
          M_Legend_Xscale(M_Legend_ori_end) +
          10,
      );
      M_Legend_rect2Labe1.attr(
        'x',
        M_Legend_Xscale(M_Legend_ori_end),
      ).text(M_Legend_end);
    }
  }

  function dragEnded2() {
    Georender();
  }

  M_Legend_rect1.call(M_Legend_drag1);
  M_Legend_rect2.call(M_Legend_drag2);

  // Leaderboard ===============================================================================
  function BarRender() {
    var data;
    if (R_region === 'all') {
      data = dataG.filter(
        (d) =>
          d.year === R_year && d.country !== 'year_average',
      );
    } else {
      data = dataG.filter(
        (d) =>
          d.year === R_year &&
          d.country !== 'year_average' &&
          d.region === R_region,
      );
    }
    R_Value_max= new Set(data.map(d => d.country)).size;
    if(MaxInput.property('value')>R_Value_max)
    {
      MaxInput.property("value", R_Value_max);
      R_Max = R_Value_max;
    }
    var R_value_max = d3.max(data, function (d) {
      return d[R_feature];
    });
    data.sort((a, b) => b[R_feature] - a[R_feature]);
    data = data.slice(R_Min - 1, R_Max);

    var R_Plot_height = Svg_height * 0.5;
    var R_Plot_width = Svg_width * 0.9;
    var R_svg = svgMain
      .append('svg')
      .attr('width', Svg_width)
      .attr('height', Svg_height)
      .append('g')
      .attr('width', R_Plot_width + 100)
      .attr('height', R_Plot_height + 100)
      .attr('transform', 'translate(50,100)');

    var R_x = d3
      .scaleBand()
      .range([0, R_Plot_width])
      .domain(
        data.map(function (d) {
          return d.country;
        }),
      )
      .padding(0.2);

    R_svg.append('g')
      .attr('transform', 'translate(0,' + R_Plot_height + ')')
      .call(d3.axisBottom(R_x))
      .selectAll('text')
      .style('font-size', '12px')
      .style('cursor','pointer')
      .attr('transform', 'translate(0,0)rotate(45)')
      .style('text-anchor', 'start')
      .on('click', function (e, d) {
        D_country = d; 
        d3.select('#D_input').property('value', D_country);
        d3.select('input[name="D_Country_Btn"][value="country"]').property('checked', true);
        d3.select('input[name="D_Data_Btn"][value="normalize"]').property('disabled', false);
        tooltip.style('visibility', 'hidden');
        d3.select('#Nav_Btn_D').dispatch('click');
      });

    // Add Y axis
    var R_y = d3
      .scaleLinear()
      .domain([0, R_value_max])
      .range([R_Plot_height, 0]);

    R_svg.append('g').call(d3.axisLeft(R_y));

    R_svg.selectAll('mybar')
      .data(data)
      .enter()
      .append('rect')
      .attr('x', function (d) {
        return R_x(d.country);
      })
      .attr('y', function (d) {
        return R_y(d[R_feature]);
      })
      .attr('width', R_x.bandwidth())
      .attr('height', function (d) {
        return R_Plot_height - R_y(d[R_feature]);
      })
      .attr('fill', D_color(R_feature))
      .on('mouseover', function (event, d) {
        d3.select(this).attr('stroke', 'red');

        const tooltipWidth = R_tooltip.node().offsetWidth;
        const maxTooltipX =
          R_svg.node().getBoundingClientRect().width - tooltipWidth -10; 
        let tooltipX = event.pageX + 10; 
        tooltipX = Math.min(tooltipX, maxTooltipX);
        const tooltipHeight = R_tooltip.node().offsetHeight;
        const maxTooltipY =
          R_svg.node().getBoundingClientRect().height - tooltipHeight -10; 

        var tooltipY = event.pageY + 10; 
        tooltipY = Math.min(tooltipY, maxTooltipY);

        R_tooltip.transition().style('opacity', 0.95);

        let tooltipContent = `<strong>${d.country}</strong><br>
                              Happiness Rank: ${d.happiness_rank}<br>
                              Happiness Score: ${d.happiness}`;

        if (R_feature !== 'happiness') {
          tooltipContent += `<br>${R_feature}: ${d[R_feature]}`;
        }

        R_tooltip.html(tooltipContent)
          .style('left', tooltipX + 'px')
          .style('top', tooltipY + 'px');
      })
      .on('mouseout', function () {
        d3.select(this).attr('stroke', 'none');
        R_tooltip.style('opacity', 0);
      });

    d3.select(document).on('click', function () {
      R_tooltip.style('opacity', 0);
    });
  }
  // Tool_R ====================================================================================
  // Btns ==================================================================================
  const R_Btns = d3.selectAll(
    '#R_Feature_Btns input[type="radio"]',
  );

  R_Btns.on('change', function () {
    R_feature = d3.select(this).attr('value');
    R_tooltip.style('opacity', 0);
    svgMain.selectAll('*').remove();
    BarRender();
  });

  // Time sliders ==========================================================================
  var R_slider = d3
    .sliderHorizontal()
    .min(2015)
    .max(2023)
    .default(2023)
    .step(1)
    .width(D_width-60)
    .displayValue(false)
    .on('onchange', () => {
      R_year = R_slider.value();
      svgMain.selectAll('*').remove();
      BarRender();
    });

  d3.select('#R_Year_Slider')
    .append('svg')
    .attr('width', D_width-25)
    .attr('height', 45)
    .append('g')
    .attr('transform', 'translate(15,10)')
    .classed('slider', true)
    .call(R_slider);

  // Region ==============================================================================
  const R_Input = d3.select('#R_input');
  const R_sug = d3.select('#R_sug');

  R_Input.on('input', function () {
    R_sug.html('');

    const inputText = this.value.toLowerCase();

    var filteredData = reg.filter((d) =>
      d.toLowerCase().startsWith(inputText),
    );
    if (inputText === "") {
      filteredData = reg;
    } 

    const slicedData = filteredData;

    if (slicedData.length > 0) {
      R_sug.style('display', 'block')
        .selectAll('.suggestion-item')
        .data(slicedData)
        .enter()
        .append('div')
        .attr('class', 'suggestion-item')
        .text((d) => d)
        .on('click', function () {
          R_Input.property('value', d3.select(this).text());
          R_sug.html('');
          R_sug.style('display', 'none');
          R_region = R_Input.property('value');
          svgMain.selectAll('*').remove();
          BarRender();
        });
    } else {
      R_sug.style('display', 'none');
    }
  });

  R_Input.on('click', function () {
    if(this.value.toLowerCase() ==""){
      R_sug.html('');
      const filteredData = reg;
      const slicedData = filteredData;
      R_sug.style('display', 'block')
        .selectAll('.suggestion-item')
        .data(slicedData)
        .enter()
        .append('div')
        .attr('class', 'suggestion-item')
        .text((d) => d)
        .on('click', function () {
          R_Input.property('value', d3.select(this).text());
          R_sug.html('');
          R_sug.style('display', 'none');
          R_region = R_Input.property('value');
          svgMain.selectAll('*').remove();
          BarRender();
        });
      }
      else{
        R_region = R_Input.property('value');
        if(reg.includes(R_region)){
          svgMain.selectAll('*').remove();
          BarRender();
        }

      }
    
    d3.select(
      'input[name="R_Country_Btn"][value="region"]',
    ).property('checked', true);
    
  });

  document.addEventListener('click', function (event) {
    const isClickInsideInputOrSuggestion =
      R_Input.node().contains(event.target) ||
      R_sug.node().contains(event.target);

    if (!isClickInsideInputOrSuggestion) {
      R_sug.style('display', 'none');
    }
  });

  d3.selectAll('input[name="R_Country_Btn"]').on(
    'change',
    function () {
      var selectedValue = d3
        .select('input[name="R_Country_Btn"]:checked')
        .node().value;
      if (selectedValue === 'all') {
        R_region = 'all';
        svgMain.selectAll('*').remove();
        BarRender();
      } else {
        R_region = R_Input.property('value');
        if(R_region!="" && reg.includes(R_region)){
          svgMain.selectAll('*').remove();
          BarRender();
        }
      }

      
    },
  );
  // Range  ==============================================================================
  var MaxInput = d3.select('#R_Max_input');
  var MinInput = d3.select('#R_Min_input');
  MinInput.on('input', function () {
    var minValue = +MinInput.property('value');
    if (minValue < 1) {
      MinInput.property('value', 1);
    } else if (minValue <= R_Max) {
      R_Min = minValue;
      svgMain.selectAll('*').remove();
      BarRender();
    } else {
      MinInput.property('value', R_Max);
    }
  });

  MaxInput.on('input', function () {
    var maxValue = +MaxInput.property('value');
    if (maxValue >= R_Value_max) {
      R_Max = R_Value_max;
      MaxInput.property('value', R_Max);
      svgMain.selectAll('*').remove();
      BarRender();
    } 
    else if (maxValue >= R_Min) {
      R_Max = maxValue;
      svgMain.selectAll('*').remove();
      BarRender();
    } else {
      MaxInput.property('value', R_Min);
    }
  });
  //Corrlation Matrix=======================================================================
  const RenderHeatmap = (data, svg, w, h) => {
    var svg_ = svg
      .append('svg')
      .attr('width', w)
      .attr('height', h)
      .attr('id', 'Svg');
    var corr = jz.arr.correlationMatrix(data, cols);
    var extent = d3.extent(
      corr
        .map(function (d) {
          return d.correlation;
        })
        .filter(function (d) {
          return d !== 1;
        }),
    );
    var grid = data2grid.grid(corr);
    var rows = cols.length;

    var inner_margin = {
      top: 30,
      bottom: 30,
      left: 100,
      right: 0,
    };
    var dim = d3.min([w , h ]);
    var inner_width = dim * 0.6,
      inner_height = dim * 0.6;

    svg_ = svg_
      .attr('width', w)
      .attr('height', h)
      .append('g')
      .attr('transform', 'translate(' + (w - inner_width) / 2 + ',' + 70 + ')');
;

    var padding = 0.1;
    var x = d3
      .scaleBand()
      .range([0, inner_width])
      .paddingInner(padding)
      .domain(d3.range(1, rows + 1));

    var y = d3
      .scaleBand()
      .range([0, inner_height])
      .paddingInner(padding)
      .domain(d3.range(1, rows + 1));

    var x_axis = d3.axisBottom(x).tickFormat(function (d, i) {
      return cols[i];
    });
    var y_axis = d3.axisLeft(y).tickFormat(function (d, i) {
      return cols[i];
    });

    var axisXcreated = svg_
      .append('g')
      .attr('class', 'x axis')
      .call(x_axis)
      .attr(
        'transform',
        'translate(0, ' + inner_height + ')',
      );
    axisXcreated
      .selectAll('text')
      .style('text-anchor', 'end')
      .attr('dx', '-0.5em')
      .attr('dy', '0.5em')
      .attr('transform', 'rotate(-45)');
    svg_.append('g').attr('class', 'y axis').call(y_axis);

    var colorscale = chroma
      .scale(['red', 'white', 'navy'])
      .domain([-1, 0, extent[1]]);

    svg_
      .selectAll('rect')
      .data(grid, function (d) {
        return d.column_a + d.column_b;
      })
      .enter()
      .append('rect')
      .attr('x', function (d) {
        return x(d.column);
      })
      .attr('y', function (d) {
        return y(d.row);
      })
      .attr('width', x.bandwidth())
      .attr('height', y.bandwidth())
      .style('fill', function (d) {
        return colorscale(d.correlation);
      })
      .transition()
      .style('opacity', 10);

    svg_
      .selectAll('rect')
      .on('mouseover', function (d) {
        var d_ = d3.select(this).datum();
        d3.select(this).classed('selected', true);       

        svg_
          .select(
            '.x.axis .tick:nth-of-type(' +
              d_.column +
              ') text',
          )
          .classed('selected', true);
        svg_
          .select(
            '.y.axis .tick:nth-of-type(' + d_.row + ') text',
          )
          .classed('selected', true);
        svg_
          .select(
            '.x.axis .tick:nth-of-type(' +
              d_.column +
              ') line',
          )
          .classed('selected', true);
        svg_
          .select(
            '.y.axis .tick:nth-of-type(' + d_.row + ') line',
          )
          .classed('selected', true);

        var [mouseX, mouseY] = d3.pointer(event);
        C_tooltip.transition().style('opacity', 0.9);

        C_tooltip.html(
          '(' +
            d_.column_x +
            ', ' +
            d_.column_y +
            ')' +
            ': ' +
            d_.correlation.toFixed(2),
        )
          .style('left', mouseX + 250 + 'px')
          .style('top', mouseY + 150 + 'px');
      })
      .on('mouseout', function () {
        svg_.selectAll('rect').classed('selected', false);
        svg_
          .selectAll('.axis .tick text')
          .classed('selected', false);
        svg_
          .selectAll('.axis .tick line')
          .classed('selected', false);

        C_tooltip.transition().style('opacity', 0);
      });
  };

  d3.select(document).on('click', function () {
    C_tooltip.style('opacity', 0);
  });

  var cols = [
    'happiness',
    'economy',
    'family',
    'health',
    'freedom',
    'trust',
    'generosity',
  ];

  //Tool_C =================================================================================
  const C_Legend_width = C_width - 60;
  const C_Legend_height = 20;

  var C_Legend_svg = d3
    .select('#C_Legend_Svg')
    .attr('width', C_Legend_width + 40)
    .attr('height', C_Legend_height + 100);

  var C_Legend_g = C_Legend_svg.append('g').attr(
    'transform',
    'translate(15, 20)',
  );

  var C_Legend_defs = C_Legend_g.append('defs');

  var C_Legend_gradient = C_Legend_defs.append(
    'linearGradient',
  ).attr('id', 'linear-gradient-C');

  var C_Legend_stops = [
    { offset: 0, color: 'red', value: C_Value_min },
    {
      offset: 0.5,
      color: 'white',
      value: (C_Value_min + C_Value_max) / 2,
    },
    { offset: 1, color: 'navy', value: C_Value_max },
  ];

  C_Legend_gradient.selectAll('stop')
    .data(C_Legend_stops)
    .enter()
    .append('stop')
    .attr('offset', function (d) {
      return 100 * d.offset + '%';
    })
    .attr('stop-color', function (d) {
      return d.color;
    });

  C_Legend_g.append('rect')
    .attr('width', C_Legend_width)
    .attr('height', C_Legend_height)
    .style('fill', 'url(#linear-gradient-C)');

  var C_Legend_Xscale = d3
    .scaleLinear()
    .domain([C_Value_min, C_Value_max])
    .range([0, C_Legend_width]);

  var C_Legend_ticks = C_Legend_g.selectAll('.tick')
    .data(C_Legend_Xscale.ticks(5)) 
    .enter()
    .append('g')
    .attr('class', 'tick')
    .attr('transform', function (d) {
      return (
        'translate(' +
        C_Legend_Xscale(d) +
        ', ' +
        C_Legend_height +
        ')'
      );
    });

  C_Legend_ticks.append('line')
    .attr('y1', 0)
    .attr('y2', 5)
    .style('stroke', 'black');

  C_Legend_ticks.append('text')
    .attr('y', 8)
    .attr('dy', '0.71em')
    .style('text-anchor', 'middle')
    .classed('legend-text', true)
    .text(function (d) {
      return d.toFixed(2);
    });

  // Detail ==================================================================================
  function LineChart() {
    var D_min;
    var D_max;
    var dataC = dataG.filter((d) => d.country === D_country);
    var dataAvg = dataG.filter((d) => d.country === D_country_C);
    var features = [
      'happiness',
      'economy',
      'family',
      'health',
      'freedom',
      'trust',
      'generosity',
    ];

    var filteredData;
    if (D_flag === 'raw') {
      filteredData = dataC;
      D_min = 0;
      D_max = d3.max(dataG, function (d) {
        return d3.max(features, function (feature) {
          return d[feature];
        });
      });
    } else {
      var outputData = [];
      for (var year = 2015; year <= 2023; year++) {
        var yearData = { year: year };
        // Iterate over features
        features.forEach(function (feature) {
          const dataCForYear = dataC.find((d) => d.year === year);
          const dataAvgForYear = dataAvg.find((d) => d.year === year);
          if (dataCForYear && feature in dataCForYear && dataAvgForYear && feature in dataAvgForYear) {
              const C = dataCForYear[feature];
              const Avg = dataAvgForYear[feature];
              yearData[feature] = +(C - Avg).toFixed(2);
          }
        });
        outputData.push(yearData);
      }
      filteredData = outputData;
      D_min = d3.min(filteredData, function (d) {
        return d3.min(features, function (feature) {
          return d[feature];
        });
      });
      D_min = Math.min(D_min, 0);
      D_max = d3.max(filteredData, function (d) {
        return d3.max(features, function (feature) {
          return d[feature];
        });
      });
      D_max = Math.max(D_max, 0);
    }
    var height = Svg_height * 0.7;
    var width = Svg_width * 0.9;
    // Create SVG element
    var margin = { top: 50, right: 40, bottom: 30, left: 40 };

    var svg = d3
      .select('#Svg_Main')
      .append('svg')
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom)
      .append('g')
      .attr(
        'transform',
        'translate(' + margin.left + ',' + margin.top + ')',
      );

    // Set up scales
    var xScale = d3
      .scaleLinear()
      .domain([2015, 2023])
      .range([0, width]);

    var yScale = d3
      .scaleLinear()
      .domain([D_min, D_max + 0.2])
      .range([height, 0]);

    var line = d3
      .line()
      .x(function (d) {
        return xScale(d.year);
      })
      .y(function (d) {
        return yScale(d[feature]);
      });

    // Add x-axis
    var xAxis = svg
        .append('g')
        .attr('transform', 'translate(0,' + height + ')')
        .call(d3.axisBottom(xScale));
    xAxis
      .selectAll('.tick')     
      .on('mouseover', function (event, d) {

        D_tooltip.transition().style('opacity', 0.9);

        var hoveredData = filteredData.find((data) => data.year === d);
        D_tooltip.html(
          'Country: ' + D_country + '<br>' +
          'Year: ' + hoveredData.year + '<br>' +
          'Happiness Rank: ' + hoveredData.happiness_rank + '<br>' +
          'Happiness: ' + hoveredData.happiness.toFixed(2)+ '<br>' +
          'Economy: ' + hoveredData.economy.toFixed(2)+ '<br>' +
          'Family: ' + hoveredData.family.toFixed(2)+ '<br>' +
          'Health: ' + hoveredData.health.toFixed(2)+ '<br>' +
          'Freedom: ' + hoveredData.freedom.toFixed(2)+ '<br>' +
          'Trust: ' + hoveredData.trust.toFixed(2)+ '<br>' +
          'Generosity: ' + hoveredData.generosity.toFixed(2)+ '<br>'
           );
    
        var maxX = Svg_width - D_tooltip.node().offsetWidth - 20; 
        var maxY = Svg_height - D_tooltip.node().offsetHeight - 65; 
        
        
        var mouseX = Math.abs(d3.pointer(event, document.body)[0]);

        D_tooltip.style('left', Math.min(mouseX,maxX) + 'px');
        D_tooltip.style('top', maxY+ 'px');
      })
      .on('mouseout', function () {
        D_tooltip.transition().style('opacity', 0);
      });
    // Add y-axis
    svg.append('g').call(d3.axisLeft(yScale));


    features.forEach(function (feature) {
      svg
        .append('path')
        .datum(filteredData)
        .attr('fill', 'none')
        .attr('stroke', D_color(feature))
        .attr('stroke-linejoin', 'round')
        .attr('stroke-linecap', 'round')
        .attr('stroke-width', 3)
        .attr(
          'd',
          line.y(function (d) {
            return yScale(d[feature]);
          }),
        );
    });
    if (D_min !== 0) {
      svg
        .append('line')
        .attr('x1', 0)
        .attr('y1', yScale(0))
        .attr('x2', width)
        .attr('y2', yScale(0))
        .attr('stroke', 'grey')
        .attr('stroke-width', 1)
        .attr('stroke-dasharray', '4');
    }
    features.forEach(function (feature) {
      svg
        .selectAll('.dot-' + feature)
        .data(filteredData)
        .enter()
        .append('circle')        
        .attr('class', 'dot-' + feature)
        .attr('cx', function (d) {
          return xScale(d.year);
        })
        .attr('cy', function (d) {
          return yScale(d[feature]);
        })
        .attr('r', 5) // You can adjust the radius of the dots
        .attr('fill', D_color(feature))
        .on('mouseover', function (event, d) {
          var [mouseX, mouseY] = d3.pointer(event);
          D_tooltip.transition().style('opacity', 0.9);
    
          D_tooltip.html(
          'Year: ' + d.year + '<br>' +
          feature + ': ' + d[feature].toFixed(2)
          );
      
          var maxX = Svg_width - D_tooltip.node().offsetWidth - 10;
          var maxY = Svg_height - D_tooltip.node().offsetHeight - 10; 
      
          D_tooltip.style('left', Math.min(mouseX + 50, maxX) + 'px');
          D_tooltip.style('top', Math.min(mouseY + 100, maxY) + 'px');
          d3.select(this).classed('highlighted-dot', true);
        })
        .on('mouseout', function () {  
          d3.select(this).classed('highlighted-dot', false);
          D_tooltip.transition().style('opacity', 0);
        });
    });

    

  }

  //Tool_D ===================================================================================
  /*Input ===================================================================================*/
  const D_Input = d3.select('#D_input');
  const D_Input2 = d3.select('#D_input2');
  const D_sug = d3.select('#D_sug');
  const D_sug2 = d3.select('#D_sug2');


  D_Input.on('input', function () {
    D_sug.html('');

    const inputText = this.value.toLowerCase();

    const filteredData = cty.filter((d) =>
      d.toLowerCase().startsWith(inputText),
    );

    const slicedData = filteredData.slice(0, 10);

    if (slicedData.length > 0) {
      D_sug.style('display', 'block')
        .selectAll('.suggestion-item')
        .data(slicedData)
        .enter()
        .append('div')
        .attr('class', 'suggestion-item')
        .text((d) => d)
        .on('click', function () {
          D_Input.property('value', d3.select(this).text());
          D_sug.html('');
          D_sug.style('display', 'none');
          D_country = D_Input.property('value');
          svgMain.selectAll('*').remove();
          LineChart();
        });
    } else {
      D_sug.style('display', 'none');
    }
  });

  D_Input2.on('input', function () {
    if(D_country!="year_average")
    {
      D_sug2.html('');

      const inputText = this.value.toLowerCase();

      const filteredData = cty_ori.filter((d) =>
        d.toLowerCase().startsWith(inputText),
      );


      const slicedData = filteredData.slice(0, 10);
      if (slicedData.length > 0) {
        D_sug2.style('display', 'block')
          .selectAll('.suggestion-item')
          .data(slicedData)
          .enter()
          .append('div')
          .attr('class', 'suggestion-item')
          .text((d) => d)
          .on('click', function () {
            D_Input2.property('value', d3.select(this).text());
            D_sug2.html('');
            D_sug2.style('display', 'none');
            D_country_C = D_Input2.property('value');
            svgMain.selectAll('*').remove();
            LineChart();
          });
      } else {
        D_sug2.style('display', 'none');
      }
    }
  });

  D_Input.on('click', function () {
    d3.select('input[name="D_Country_Btn"][value="country"]').property('checked', true);
    d3.select('input[name="D_Data_Btn"][value="normalize"]').property('disabled', false);
    D_country = d3.select('#D_input').property('value');
    svgMain.selectAll('*').remove();
    LineChart();
  });

  D_Input2.on('click', function () {
    if(D_country!="year_average")
    {
      d3.select('input[name="D_Data_Btn"][value="normalize"]').property('disabled', false);
      d3.select('input[name="D_Data_Btn"][value="normalize"]').property('checked', true);
      D_flag = "average";
      D_country_C = d3.select('#D_input2').property('value');
      svgMain.selectAll('*').remove();
      LineChart();
    }    
  });

  const D_Btn = d3.select('#D_Btn');
  D_Btn.on('click', function () {
    d3.select('#Nav_Btn_M').dispatch('click');

  });

  document.addEventListener('click', function (event) {
    const isClickInsideInputOrSuggestion =
      D_Input.node().contains(event.target) ||
      D_sug.node().contains(event.target);

    if (!isClickInsideInputOrSuggestion) {
      D_sug.style('display', 'none');
    }
  });



  /*Country radio===========================================================================*/
  const D_Country_Btns = d3.selectAll(
    '#D_Country_Btns input[type="radio"]',
  );

  D_Country_Btns.on('change', function () {
    var flag = d3
      .select('input[name="D_Country_Btn"]:checked')
      .node().value;
    if (flag == 'average') {
      D_country = 'year_average';
      D_flag= 'raw';
      d3.select('input[name="D_Data_Btn"][value="raw"]').property('checked', true);
      d3.select('input[name="D_Data_Btn"][value="normalize"]').property('disabled', true);
      svgMain.selectAll('*').remove();
      LineChart();
    } else {
      d3.select('input[name="D_Data_Btn"][value="normalize"]').property('disabled', false);
      D_country = d3.select('#D_input').property('value');
      svgMain.selectAll('*').remove();
      LineChart();
    }
  });

  const D_Data_Btns = d3.selectAll(
    '#D_Data_Btns input[type="radio"]',
  );

  D_Data_Btns.on('change', function () {
    D_flag = d3
      .select('input[name="D_Data_Btn"]:checked')
      .node().value;
    svgMain.selectAll('*').remove();
    LineChart();
  });

  // Legend ==============================================================================
  var D_Legeng_svg = d3.select('#D_Legend_Svg');

  var D_Legend_size = 15;
  D_Legeng_svg.selectAll('D_rects')
    .data(D_keys)
    .enter()
    .append('rect')
    .attr('x', 10)
    .attr('y', function (d, i) {
      return 10 + i * (D_Legend_size + 5);
    })
    .attr('width', D_Legend_size)
    .attr('height', D_Legend_size)
    .style('fill', function (d) {
      return D_color(d);
    });

  D_Legeng_svg.selectAll('D_Legend_labels')
    .data(D_keys)
    .enter()
    .append('text')
    .attr('x', 10 + D_Legend_size * 1.2)
    .attr('y', function (d, i) {
      return 10 + i * (D_Legend_size + 5) + D_Legend_size / 2;
    })
    .style('fill', function (d) {
      return D_color(d);
    })
    .text(function (d) {
      return d;
    })
    .attr('text-anchor', 'left')
    .style('alignment-baseline', 'middle');


}(d3, topojson));
