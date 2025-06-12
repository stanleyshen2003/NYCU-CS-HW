#include <iostream>
#include <iomanip>
#include <cmath> 
using namespace std;
//p>=0, 0<=r<=20, -20<=p<=0, t<=0
//decreasing
double f(double x,double p,double q,double r,double s,double t,double u){
	return p*exp(-x) + q*sin(x) + r*cos(x) + s*tan(x) + t*x*x +u;
}
int main(){
	double high,low,middle,p,q,r,s,t,u;
	while(cin>>p>>q>>r>>s>>t>>u){
		if(f(1,p,q,r,s,t,u)*f(0,p,q,r,s,t,u)>0){
			cout<<"No solution"<<endl;
			continue;
		}
		high=1,low=0;
		while(high-low>=1e-10){
			middle=(high+low)/2;
			if(f(middle,p,q,r,s,t,u)>=0){
				low=middle;
			}
			else high=middle;
		}
		cout<<fixed<<setprecision(4)<<middle<<endl;
	} 

	return 0;
}

