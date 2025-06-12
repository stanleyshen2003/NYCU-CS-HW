(function () {
  'use strict';

  // set margin outside
  var last1, last2;
  var margin = {top: window.innerHeight/10, right: 10, bottom: window.innerHeight/3, left: 80},
      width = window.innerHeight - margin.left - margin.right,
      height = window.innerHeight - margin.top - margin.bottom;
  var dataGlobal;
  // append the svg object to the body of the page
  var svg = d3.select("#svg")
    .append("svg")
      .attr("width", width + margin.left + margin.right)
      .attr("height", height + margin.top + margin.bottom)
    .append("g")
      .attr("transform",
            "translate(" + margin.left + "," + margin.top + ")");

  // render function
  const render = (data, sort_by, descend, low, high) =>{
    last1 = low;
    last2 = high;
  	svg.selectAll("*").remove();
    data = descend? data.sort((a,b) => b[sort_by] - a[sort_by]) : data.sort((a,b) => a[sort_by] - b[sort_by]);
    data = data.filter(function(d,index){
    	return index>=low && index<=high;
    });
    
    var subgroups = ['teaching_score', 'research_score', 'citations_score', 'industry_income_score', 'international_score'];
    if (subgroups[0] != sort_by && sort_by!='total_score'){
      var index = subgroups.indexOf(sort_by);
    	var temp = subgroups[0];
  		subgroups[0] = sort_by;
      subgroups[index] = temp;
    }
  	var groups = d3.map(data, function(d){return(d.name)});
    // console.log(groups)
    var x = d3.scaleBand()
        .domain(groups)
        .range([0, width])
        .padding([0.2]);
    svg.append("g")
      .attr("transform", "translate(0," + height + ")")
      .call(d3.axisBottom(x).tickSizeOuter(0))
      .selectAll("text")  // Select all text elements within the axis
      .style('text-anchor', 'end')
      .attr('dx', '-0.7em')
      .attr('dy', '0.5em')
      .attr('transform', 'rotate(-45)');

    // Add Y axis
    var y = d3.scaleLinear()
      .domain([0, 100])
      .range([ height, 0 ]);
    svg.append("g")
      .call(d3.axisLeft(y));

    // color palette = one color per subgroup
    var color = d3.scaleOrdinal()
      .domain(['teaching_score', 'research_score', 'citations_score', 'industry_income_score', 'international_score'])
      .range(['#e41a1c','#377eb8','#4daf4a','#FFFF00', '#FF00FF']);

    //stack the data? --> stack per subgroup
    var stackedData = d3.stack()
      .keys(subgroups)
      (data);

    // Show the bars
    svg.append("g")
      .selectAll("g")
      // Enter in the stack data = loop key per key = group per group
      .data(stackedData)
      .enter().append("g")
        .attr("fill", function(d) { return color(d.key); })  
        .selectAll("rect")
        // enter a second time = loop subgroup per subgroup to add all rectangles
        .data(function(d) { return d; })
        .enter().append("rect")
          .attr("x", function(d) { return x(d.data.name); })
          .attr("y", function(d) { return y(d[1]); })
          .attr("height", function(d) { return y(d[0]) - y(d[1]); })
          .attr("width",x.bandwidth())
    		.on("mouseover", function(event,d) {
          showTooltip(d, sort_by);
          tooltip.style("left", (event.pageX + 10) + "px");
        	tooltip.style("top", (event.pageY - 10) + "px");

        })
    .on("mouseout", hideTooltip);
  	
  };
  var tooltip = d3.select("body").append("div")
    .attr("class", "tooltip")
    .style("opacity", 0);

  function showTooltip( d, sortby) {
    console.log();
    const name = d.data.name;
    tooltip.html(`
    <p>Name: ${name}</p>
    <p>Total Score: ${d.data.total_score}</p>
    <p>Selected Rank: ${sortby}</p>
    <p>Selected Score: ${d.data[sortby]}</p>

  `);

    // Position the tooltip near the mouse pointer
    
    tooltip.style("opacity", 1);
  }

  function hideTooltip() {
    tooltip.style("opacity", 0);
  }


  d3.csv('https://raw.githubusercontent.com/stanleyshen2003/Data-Visualization/main/HW5%20-%20stacked%20bar%20chart/TIMES_WorldUniversityRankings_2024.csv').then(data => {
    const weights = {
      teaching: 0.295,
      research: 0.29,
      citation: 0.30,
      industry: 0.04,
      international: 0.075
    };
    data = data.filter(function(d) {
      return d.rank !== "Reporter";
    });
  	data = data.map(function(d) {
    	return {
    		name: d.name,
        teaching_score: (+d.scores_teaching * weights["teaching"]).toFixed(2),
        teaching_rank: +d.scores_teaching_rank,
        research_score: (+d.scores_research * weights["research"]).toFixed(2),
        research_rank: +d.scores_research_rank,
        citations_score: (+d.scores_citations * weights["citation"]).toFixed(2),
        citations_rank: +d.scores_citations_rank,
        industry_income_score: (+d.scores_industry_income * weights["industry"]).toFixed(2),
        industry_income_rank: +d.scores_industry_income_rank,
        international_score: (+d.scores_international_outlook * weights["international"]).toFixed(2),
        international_rank: +d.scores_international_outlook_rank,
        total_score: +(+d.scores_teaching * weights["teaching"] + +d.scores_research * weights["research"] +
          +d.scores_citations * weights["citation"] + +d.scores_industry_income * weights["industry"]
        + +d.scores_international_outlook * weights["international"]).toFixed(1)
      };
    });
    data.sort((a,b)=>b.total_score - a.total_score);
    data.forEach((d, index)=>{
    	d.total_rank = index+1; 
    });
    dataGlobal = data;
    
    render(data, "total_score", 1, 1, 20);
  });

  function getOption(name){
    var options = document.getElementsByName(name);
    var selectedX;
    options.forEach(option => {
      if (option.checked) {
        selectedX = option.value;
      }
    });
    return selectedX;
  }

  function getNum(name){
    var element = document.getElementById(name);
    //console.log(element);
    return element.value;
  }

  function renderall(){
    var option1 = getOption('options');
    var option2 = +(getOption('options2'));
    var min = +(getNum('head'));
    var max = +(getNum('tail'));
    if(min > max){
    	document.getElementById('head').value = last1;
      document.getElementById('tail').value = last2;
      return
    }
    // console.log(min);
    // console.log(max);
    render(dataGlobal, option1, option2,min,max);
  	// console.log(option1);
  }

  document.addEventListener('DOMContentLoaded', function() {
    const radiobotx = document.getElementsByName('options');
    const radioboty = document.getElementsByName('options2');
    const range1 = document.getElementById('head');
    const range2 = document.getElementById('tail');
    //console.log(radiobotx);
    //console.log(radiobotx.length);

    // Add event listeners to radio buttons here
    radiobotx.forEach(radioButton => {
      radioButton.addEventListener('change', renderall);
      //console.log('added');
    });
    radioboty.forEach(radioButton => {
      radioButton.addEventListener('change', renderall);
      //console.log('added');
    });
    
    
    range1.addEventListener('change', renderall);
    
  	
    range2.addEventListener('change', renderall);

    
  });

}());
