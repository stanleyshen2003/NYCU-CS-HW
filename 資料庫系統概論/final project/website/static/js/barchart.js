//圖表配置
var option3={
    chart:{
        type: 'bar',
        backgroundColor: '#f6faff',
        style:{
            
            color:'#9eb9df',
        }
    },
    title:{
        text:""
    },
    credits:{
        enabled:false
    },
    tooltip:{
        borderColor:'#eee3c3',
        borderRadius:20,
        style:{
            color:'#a9c2e4'
        },
        formatter: function () {
            return this.x +" "+this.y+"%";
        }
    },
    legend:{
        itemStyle: {
            color: '#9eb9df',
            fontWeight: 'bold',
            fontSize: '16px'
         }
    },
    colors:['#e2d5ac'],
    xAxis:{
        categories:["C++","HTML","CSS","JavaScript"],
        labels: {
            style: {
               color: '#9eb9df',
               fontWeight: 'bold',
                fontSize: '14px'
            }
         }
    },
    yAxis: {
        min: 0,
        title: {
            text: ''
        },
        labels: {
            overflow: 'justify'
        }
    },
    //圖表數據
    series:[
        {
            color:'#9eb9df',
            name:"language",
            data:[80,70,50,20]           
        }
    ]
};
//顯示圖表初始化
Highcharts.chart("container3",option3);