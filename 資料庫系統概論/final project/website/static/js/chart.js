//圖表配置
var option1={
    chart:{
        marginTop:60,
        backgroundColor: '#f6faff',
        style:{
            
            color:'#e2d5ac'
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
        categories:["一上","一下","二上"],
        labels: {
            style: {
               color: '#dacb9c',
               fontWeight: 'bold',
                fontSize: '14px'
            }
         }
    },
    yAxis:{
        softMax:5,
        endOnTick:false,
        softMin:0,
        minorTickInterval: 0.1,
        labels: {
            style: {
               color: '#dfd0a3',
               fontWeight: 'bold',
                fontSize: '14px'
            }
         },
        title:{
            textAlign: 'right',
            rotation: 0,
            x: 20,
            y: -100,
            text:"GPA",
            style: {
                color: '#9eb9df',
                fontWeight: 'bold',
                 fontSize: '16px'
             }
        }
    },
    //圖表數據
    series:[
        {
            name:"GPA",
            data:[4.27,4.24,4.27]            
        }
    ]
};
//顯示圖表初始化
Highcharts.chart("container1",option1);
