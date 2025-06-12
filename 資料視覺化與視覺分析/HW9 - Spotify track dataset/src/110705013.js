var dataG;
var genreG;

var margin = {top: 0, right: 0, bottom: 0, left: 0},
    width = window.innerWidth -margin.left - margin.right,
    height = window.innerHeight -margin.top - margin.bottom;

var svgContainer = d3.select("#svg-container");

var headInput = d3.select("#head");
var head = +headInput.property("value");
var tailInput = d3.select("#tail");
var tail = +tailInput.property("value");
var bar_flag = "S";
var bar_gen_flag= "all";
var bar_art_flag="all";
var ana_flag = "P";

//排名=====================================================================
const Leaderboard = () =>{
  var N_h = d3.select("#NavigationBar").node().getBoundingClientRect().height;
  var h_tmp = height-N_h*3
  var w_tmp = width*0.96
  
  var svg1 = appendChartDiv(w_tmp, h_tmp);
  
  var data_draw = dataG
  var data_tmp = data_draw.map(function(d) {
      return {
        artists: d.artists,
        album_name: d.album_name,
        track_name: d.track_name,
        track_genre:d.track_genre,
        popularity: d.popularity
      };
    });
  if(bar_flag=='S')
  {
    RenderBarChart(svg1,data_tmp, head, tail,"track_name","popularity", w_tmp, h_tmp)
  }
 	else if( bar_flag=='A')
  {
    var data_artist = d3.rollup(data_tmp, v => d3.mean(v, d => +d.popularity), d => d.artists);
    data_artist = Array.from(data_artist, ([key, value]) => ({ artists: key, popularity: value }));
    RenderBarChart(svg1,data_artist, head, tail,"artists","popularity", w_tmp, h_tmp)
  }
  
  else if( bar_flag=='G')
  {
    var data_genre = d3.rollup(data_tmp, v => d3.mean(v, d => +d.popularity), d => d.track_genre);
  	data_genre = Array.from(data_genre, ([key, value]) => ({ genre: key, popularity: value }));
    RenderBarChart(svg1,data_genre, head, tail,"genre","popularity", w_tmp, h_tmp)
  }
  else if(bar_flag=="SOG")
  {
     data_draw= data_tmp.filter(function(d) {
        return d.track_genre === bar_gen_flag;
    	});
    RenderBarChart(svg1,data_draw, head, tail,"track_name","popularity", w_tmp, h_tmp)
    	
  }
  else if(bar_flag=="SOA")
  {
     data_draw= data_tmp.filter(function(d) {
        return d.artists=== bar_art_flag;
    	});
    	console.log(data_draw)
    	RenderBarChart(svg1,data_draw, head, tail,"track_name","popularity", w_tmp, h_tmp)
  } 
  
  headInput.on("input", function () {
    	d3.select("#Leaderboard").property("checked", true);
      if (+headInput.property("value") <= tail) {
          head = +headInput.property("value");
          svgContainer.selectAll('*').remove();
          Leaderboard();
      } else {
          headInput.property("value", tail);
      }
  });

  tailInput.on("input", function () {    
    	d3.select("#Leaderboard").property("checked", true);
      if (+tailInput.property("value") >= head) {
          tail = +tailInput.property("value");
          svgContainer.selectAll('*').remove();
          Leaderboard();
      } else {
          tailInput.property("value", head);
      }
  });
  
}

function generateIndex(averages) {
  // 对 averages 数组进行排序，按照值的大小升序排列
  const sortedAverages = averages.slice().sort((a, b) => a.value - b.value);

  // 构建排名对象数组
  const ranks = sortedAverages.map((item, index) => {
    return {
      axis: item.axis,
      rank: index + 1
    };
  });

  const order = ['danceable', 'energy', 'speech', 'acoustic', 'instrumental', 'live', 'valence'];
  const idx = order.map(attr => String(ranks.find(item => item.axis === attr).rank)).join('');

  return +idx;
}

var averagesG;
var averagesP;
var averagesI;
function initAverage(){
  const groupedData = d3.group(dataG, (d) => d.track_genre);
  var Averages = [];
  groupedData.forEach((genreData, genre, index) => {
    const averages = [];
    const avgDanceability = d3.mean(genreData, (d) => d.danceability);
    const avgEnergy = d3.mean(genreData, (d) => d.energy);
    const avgSpeechiness = d3.mean(genreData, (d) => d.speechiness);
    const avgAcousticness = d3.mean(genreData, (d) => d.acousticness);
    const avgInstrumentalness = d3.mean(genreData, (d) => d.instrumentalness);
    const avgLiveness = d3.mean(genreData, (d) => d.liveness);
    const avgValence = d3.mean(genreData, (d) => d.valence);
    const avgPopularity = d3.mean(genreData, (d) => d.popularity).toFixed(2);

    averages.push(
      { axis: 'danceable', value: avgDanceability },
      { axis: 'energy', value: avgEnergy },
      { axis: 'speech', value: avgSpeechiness },
      { axis: 'acoustic', value: avgAcousticness },
      { axis: 'instrumental', value: avgInstrumentalness },
      { axis: 'live', value: avgLiveness },
      { axis: 'valence', value: avgValence }
    );
    var idx =  generateIndex(averages);
    
    Averages.push({
      gen: genre,
      pop: avgPopularity,
      data: [averages],
      index:idx
    })
  });  
 	averagesG = Averages.slice().sort((a, b) => a.gen.localeCompare(b.gen));
  averagesP = Averages.slice().sort((a, b) => b.pop - a.pop);
  averagesI = Averages.slice().sort((a, b) => b.index - a.index);
}



// 音樂分析==================================================================
function AnalysisAll() {
  // const groupedData = d3.group(dataG, (d) => d.track_genre);
  
  const numCols = 8;
  // const numRows = Math.ceil(dataG.size / numCols);

  const divWidth = window.innerWidth * 0.96 / (numCols);
	// const divArray = [];
  
  if(ana_flag==='G')
  {
  	Averages = averagesG;
  }
  else if (ana_flag==='P')
  {
  	Averages = averagesP;
  }
  else if (ana_flag==='I')
  {
  	Averages = averagesI;
  }
  for (const genre in Averages) {
    const div = svgContainer
      .append("div")
      .style("width", divWidth + "px")
      .style("display", "inline-block")
      .attr("class", "radar-chart-div");
    
    RenderRadarChart(
      div.node(),
      Averages[genre].data, 
      Averages[genre].gen,     
      Averages[genre].pop,    
      divWidth
    );
  }
  
	
}
// HeatMap================================================================================
var cols = ['popularity','explicit','acousticness','danceability','energy','instrumentalness','liveness',
      'loudness','speechiness','valence','tempo','mode','key','duration_ms'];
function CorrelationMatrix(){
	var w_tmp = width*0.98
  var Lgd = appendChartDiv(width, height*0.1);
  var h_tmp = height*0.8
  var Svg = appendChartDiv(width, h_tmp);
  
  RenderHeatmap(dataG, Svg,w_tmp, h_tmp);
  HeatLegend(Lgd,w_tmp,height*0.1);
}

// Helper function to append a div to the container
function appendChartDiv(w,h) {
    var svg_ = svgContainer 
      .append('div')
      .style('width', w*0.98+'px')
      .style('height', h+ 'px')
      .style('float', 'left');
  	return svg_
}

//讀資料=====================================================================
d3.csv('https://raw.githubusercontent.com/stanleyshen2003/Data-Visualization/main/HW9%20-%20Spotify%20track%20dataset/dataset.csv').then(data => {
  data = data.map(function(d) {
    return {
      artists: d.artists,
      album_name: d.album_name,
      track_name: d.track_name,
      track_genre: d.track_genre,
      popularity: +d.popularity,
      explicit: d.explicit === "True" ? 1 : 0,
      acousticness: +d.acousticness,
      danceability: +d.danceability,
      energy: +d.energy,
      instrumentalness: +d.instrumentalness,
      liveness: +d.liveness,
      loudness: +d.loudness,
      speechiness: +d.speechiness,
      valence: +d.valence,
      tempo: +d.tempo,
      mode: +d.mode,
      key: +d.key,
      time_signature: d.time_signature,
      duration_ms: +d.duration_ms
  	};
	});
  dataG = data;
  genreG = [...new Set(dataG.map(d => d.track_genre))];
  artistG = [...new Set(dataG.map(d => d.artists))];
  svgContainer.selectAll("*").remove();
  Leaderboard();
  setcorr(dataG, cols)
  initAverage();
});
  


  
      
      