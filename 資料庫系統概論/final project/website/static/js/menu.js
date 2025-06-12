function openMenu(number)
{
    var menu=document.getElementById('menu'+number);
    var menu_bg=document.getElementById('menu_bg'+number);
    menu.classList.toggle("hide");
    if(parseInt(number)<=4)
    {
        menu_icon.classList.toggle("rotate1");
        menu_bg.classList.toggle("change_radius");
    }
    else
    {
        menu_icon.classList.toggle("rotate2");
    } 
    if(parseInt(number)==6)
    {
        Highcharts.chart("container1",option1);
    }
    if(parseInt(number)==7)
    {
        Highcharts.chart('container3',option3);
    }  
    if(parseInt(number)==8)
    {
        Highcharts.chart('container2',option2);
    }    
}