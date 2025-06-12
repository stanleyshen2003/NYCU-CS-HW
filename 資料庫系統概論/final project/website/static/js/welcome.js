let intro=document.querySelector('.intro');
let logo = document.querySelector('.logo-header');
let logoSpan = document.querySelectorAll('.logo');
let fade = document.querySelectorAll('.fade_in');
let slide_l = document.querySelectorAll('.slide_in_l'); 
let slide_r = document.querySelectorAll('.slide_in_r'); 

window.addEventListener('DOMContentLoaded',()=>{
    setTimeout(()=>{
        logoSpan.forEach((span,idx)=>{
            setTimeout(()=>{
                span.classList.add('active');
            },(idx+1)*400)
        });
        setTimeout(()=>{
            logoSpan.forEach((span)=>{
                setTimeout(()=>{
                    span.classList.remove('active');
                    span.classList.add('fade');
                })
            })
        },2000)

        setTimeout(()=>{
            intro.style.top='-100vh';
            intro.style.zIndex='0';
            
        },2300)
        setTimeout(()=>{
            fade.forEach((span)=>{
                setTimeout(()=>{
                    span.classList.add('appear');
                })
            })
        },2650)
        setTimeout(()=>{
            slide_l.forEach((span,idx)=>{
                setTimeout(()=>{
                    span.classList.add('appear');
                },(idx+1)*200)
            })
        },2650)
        setTimeout(()=>{
            slide_r.forEach((span,idx)=>{
                setTimeout(()=>{
                    span.classList.add('appear');
                },(idx+1)*200)
            })
        },2650)
    })
})