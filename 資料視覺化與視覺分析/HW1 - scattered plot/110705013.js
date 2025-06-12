(function (d3) {
  'use strict';

  const svg = d3.select('svg');

  const width = +svg.attr('width');
  const height = +svg.attr('height');

  const render = (data,first,second) => {
    // clear old svg
    svg.selectAll('*').remove();
    // choose columns
    const xValue = d => d[first];
    const yValue = d => d[second];
    const innerWidth = 480;
    const innerHeight = 300;
    
    // set scaler for x and y axes
    const xScale = d3.scaleLinear()
    	.domain([d3.min(data, xValue),d3.max(data, xValue)])
    	.range([0,innerWidth]);
    
    const yScale = d3.scaleLinear()
    	.domain([d3.min(data,yValue), d3.max(data, yValue)])
    	.range([innerHeight,0]);
  	
    // create a graph
    const windowWidth = window.innerWidth*0.34 - 220;
  	const windowHeight = window.innerHeight/2 - 150;
    const g = svg.append('g')
    	.attr('transform', `translate(${windowWidth},${windowHeight})`);
  	
    // set tick format for x axis
    const xAxisTickFormat = number =>
    	d3.format('.3s')(number);
    
    const yAxisTickFormat = number =>
    	d3.format('.3s')(number);
    
    // create x axis
    const xAxis = d3.axisBottom(xScale)
    	.tickFormat(xAxisTickFormat)
    	.tickSize(-innerHeight);
    
    const yAxis = d3.axisLeft(yScale)
    	.tickFormat(yAxisTickFormat)
    	.tickSize(-innerWidth);
    /*
    g.append('g')
    	.call(axisLeft(yScale))
    	.selectAll('domain, .tick line')
    	.remove(); */
    
    const xAxisG = g.append('g').call(xAxis)
    	.attr('transform', `translate(0,${innerHeight})`);
    const yAxisG = g.append('g').call(yAxis)
    	.attr('transform', `translate(0,0)`);
    
    xAxisG.select('.domain').remove();
    yAxisG.select('.domain').remove();
    
    // add axes label
    xAxisG.append('text')
    	.attr('class', 'axis-label')
    	.attr('y',40)
    	.attr('x',innerWidth/2)
    	.attr('fill', 'black')
    	.text(first);
    
    yAxisG.append('text')
    	.attr('class', 'axis-label')
    	.attr('transform', 'rotate(-90)')
    	.attr('y', -40)
    	.attr('x', -110) 
    	.style('text-anchor', 'left')
    	.attr('fill', 'black')
    	.text(second);
    
    
    // draw circles
    g.selectAll('circle').data(data)
    	.enter().append('circle')
    	.attr('cy', d => yScale(yValue(d)))
  		.attr('cx', d => xScale(xValue(d)))
    	.attr('r', 5)
      .attr('fill', d => {
        if (d.class == 'Iris-setosa') {
          return '#259518';
        } 
        else if (d.class == 'Iris-versicolor') {
          return '#106155';
        }
        else {
         return '#050934'; 
        }
      });
    
    // add title
    g.append('text')
    	.attr('class', 'title')
    	.attr('y', -15)
    	.text('scattered plot');
  };
  var dataGlobal;
  const columnsToInt = ['sepal length', 'sepal width', 'petal length', 'petal width'];
  d3.csv('https://raw.githubusercontent.com/stanleyshen2003/Data-Visualization/main/HW1%20-%20scattered%20plot/iris.csv').then(data => {
  	data.forEach(d => {
      columnsToInt.forEach(column => {
        d[column] = +d[column];
      });
    });
    console.log(data);
    data.pop();
    render(data,'sepal length', 'sepal width');
    dataGlobal = data;
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

  function renderall(){
    var option1 = getOption('options');
    var option2 = getOption('options2');
    render(dataGlobal, option1, option2);
  	//console.log(option1);
  }

  //console.log(getOption('options'));

  /*
  const radiobotx = document.getElementsByName('options');
  console.log(radiobotx);
  console.log(radiobotx.length);*/
  document.addEventListener('DOMContentLoaded', function() {
    const radiobotx = document.getElementsByName('options');
    const radioboty = document.getElementsByName('options2');
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
  });

}(d3));

