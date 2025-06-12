(function () {
  'use strict';

  var dataG = [];
  var year;
  const year_map = {
    "2017": 0,
    "2018": 1,
    "2019": 2
  };
  const pollution_map = {
    "SO2": 0,
    "NO2": 1,
    "CO": 2,
    "O3": 3,
    "PM10": 4,
    "PM2.5": 5
  };

  var bands = 2;
  const color_scheme = [
    d3.schemeYlOrBr,
    d3.schemeReds,
    d3.schemePurples,
    d3.schemeGreens,
    d3.schemeOrRd,
    d3.schemeGreys
  ];
  const width = window.innerWidth;
  var Svg = {};
  for (var i = 0; i < 6; i++) {
    Svg[i] = d3
      .select('#svg' + (i+1))
      .attr('width', width)
      .attr('height', 25*25+50+25)
      .append('g');
  }

  const marginRight = 15;
  const marginLeft = 30;
  const marginTop = 70;
  const marginBottom = 0;



  function render(year,i, svg, scheme,  pollution, filter) {
    // Construct scales and axes.
    var data = dataG[year_map[year]][pollution_map[pollutantNames[i]]];
    if(filter && i>3){

      data = data.filter(d => d.value < 100);
    };
    if(filter && i==2){
      data = data.filter(d => d.value<3);
    }
    const series = d3.rollup(data, (values, i) => d3.sort(values, d => d.date), d => d.station);
    // Specify the dimensions of the chart.

    const size = 25; // height of each band.
    const height = series.size * size + marginTop + marginBottom; // depends on the number of series
    const padding = 1;
    const colors = scheme[Math.max(3, bands)].slice(Math.max(0, 3 - bands));
    
    svg.append('text')
     .text(pollution)
     .attr('x', 10)  // Adjust the X position
     .attr('y', 23.5)  // Adjust the Y position
     .attr('font-size', '15px')  // Set the font size
     .attr('fill', 'black')
    	.style("font-weight", "bold");
    // Create the horizontal (temporal) scale.
    const x = d3.scaleUtc()
      .domain(d3.extent(data, d => d.date))
      .range([marginLeft, width - marginRight - marginLeft]);

    // Create the vertical scale; it describes the ?�total??height of the area,
    // when bands are not superimposed. The area shape will start from the y=size position
    // to represent 0 up to *bands* times the maximum band height.
    const y = d3.scaleLinear()
      .domain([0, d3.max(data, d => d.value)])
      .range([size, size - bands * (size - padding)]);

    const area = d3.area()
      .defined(d => !isNaN(d.value))
      .x((d) => x(d.date))
      .y0(size)
      .y1((d) => y(d.value));

    // A unique identifier (to avoid conflicts) for the clip rect and the reusable paths.
    const uid = `O-${Math.random().toString(16).slice(2)}`;

    // Create the SVG container.
    // svg 
    //     .attr("viewBox", [0, 0, width, height])
    //     .attr("style", "max-width: 100%; height: auto; font: 10px sans-serif;");

    // Create a G element for each location.
    const g = svg.append("g")
      .selectAll("g")
      .data(series)
      .join("g")
        .attr("transform", (d, i) => `translate(0,${i * size + marginTop})`);

    // Add a rectangular clipPath and the reference area.
    const defs = g.append("defs");
    defs.append("clipPath")
        .attr("id", (_, i) => `${uid}-clip-${i}`)
      .append("rect")
        .attr("y", padding)
        .attr("width", width-marginRight-marginLeft)
        .attr("height", size - padding);

    defs.append("path")
      .attr("id", (_, i) => `${uid}-path-${i}`)
      .attr("d", ([, values]) => area(values));

    // Create a group for each location, in which the reference area will be replicated
    // (with the SVG:use element) for each band, and translated.
    g.append("g")
      .attr("clip-path", (_, i) => `url(${new URL(`#${uid}-clip-${i}`, location)})`)
      .selectAll("use")
      .data((_ ,i) => new Array(bands).fill(i))
      .enter().append("use")
        .attr("xlink:href", (i) => `${new URL(`#${uid}-path-${i}`, location)}`)
        .attr("fill", (_, i) => colors[i])
        .attr("transform", (_, i) => `translate(0,${i * size})`);

    // Add the labels.
    g.append("text")
        .attr("x", 4)
        .attr("y", (size + padding) / 2)
        .attr("dy", "0.35em")
        .text(([name]) => name)
        .attr('font-size', '12px');

    // Add the horizontal axis.
    svg.append("g")
      .attr("transform", `translate(0,${marginTop})`)
      .call(d3.axisTop(x).ticks((width-marginRight-marginLeft) / 80).tickSizeOuter(0))
      .call(g => g.selectAll(".tick").filter(d => x(d) < marginLeft || x(d) >= width - marginRight).remove())
      .call(g => g.select(".domain").remove());
    
    //===============================================================================
    

  }
  var pollutantNames = ["SO2", "NO2", "CO", "O3", "PM10", "PM2.5"];
  d3.csv('https://raw.githubusercontent.com/stanleyshen2003/Data-Visualization/main/HW7%20-%20horizon%20chart/air-pollution.csv').then(function (datain) {
    // group them according to date and station
    datain = d3.nest()
      .key(function(d) { return d["Measurement date"].split(" ")[0] + d["Station code"]; })
      .rollup(function(d) {
        return {
          'time': new Date(d[0]['Measurement date'].split(" ")[0]),
          'Station code': d[0]['Station code'],
          'SO2': +(d3.median(d, function(d){return parseFloat(d['SO2']); })).toFixed(4),
          'NO2': +(d3.median(d, function(d){return parseFloat(d['NO2']); })).toFixed(4),
          'O3': +(d3.median(d, function(d){return parseFloat(d['O3']); })).toFixed(4),
          'CO': +(d3.median(d, function(d){return parseFloat(d['CO']); })).toFixed(4),
          'PM10': +(d3.median(d, function(d){return parseFloat(d['PM10']); })).toFixed(4),
          'PM2.5': +(d3.mean(d, function(d){return parseFloat(d['PM2.5']); })).toFixed(4)
        };
      })
      .entries(datain);
  	
    datain = datain.map(function(item) {
      return item.value;
    });
    datain = datain.filter(row => pollutantNames.every(column => row[column] >= 0));
  	const years = [2017, 2018, 2019];

    
    years.forEach(year => {
      const yearData = datain.filter(item => item.time.getFullYear() === year);
      const yearDataModified = pollutantNames.map(property => {
        return yearData.map(d => ({
          value: d[property],
          date: d.time,
          station: d["Station code"]
        }));
      });
      dataG.push(yearDataModified);
    });
  	year = "2017";
    //render("2017",pollutantNames[0]);
    for (var i = 0; i < 6; i++) {
     	render(year,i,Svg[i],color_scheme[i],pollutantNames[i], true);
    }
    console.log(dataG)
  	
  });

  // ...
  function renderALL(){
  	  var yearn = document.getElementById("fruit-select").value;
    	var band = document.getElementById("bands").value;
      var check = document.getElementById("myCheckbox").checked;

    	console.log(yearn);
    	console.log(band);
    	bands = +band;

    	for(var i=0;i<6;i++){
        Svg[i].selectAll("*").remove();
        render(yearn, i, Svg[i], color_scheme[i], pollutantNames[i], check);
      }
    
    
  }
  // Add an event listener to the "fruit-select" dropdown
  document.getElementById("fruit-select").addEventListener("change", renderALL);
  document.getElementById("bands").addEventListener("change", renderALL);
  document.getElementById("myCheckbox").addEventListener("change",renderALL)
  // // Initial rendering for the default year
  // var initialYear = "2017"; // You can change this to any default year you prefer
  // render(initialYear, dataG[year_map[initialYear]][pollution_map[pollutantNames[0]]], Svg[0], color_scheme[0], pollutantNames[0]);

}());
