#include <bits/stdc++.h>
using namespace std;
struct point{
    int x;
    int y;
};

bool cmp(point pt1, point pt2){
    return (pt1.y < pt2.y) || (pt1.y == pt2.y && pt1.x < pt2.x);
}

int cross(point pt1, point pt2, point pt3){
    return (pt2.x - pt1.x) * (pt3.y - pt1.y) - (pt2.y - pt1.y) * (pt3.x - pt1.x);
}

int l2(point pt1, point pt2){
    return (pt1.x - pt2.x) * (pt1.x - pt2.x) + (pt1.y - pt2.y)*(pt1.y - pt2.y);
}
point pt_ini;
bool comp_angle(point pt1, point pt2){
    int ang = cross(pt_ini, pt1, pt2);
    return (ang > 0) || (ang == 0 && l2(pt_ini, pt1) < l2(pt_ini, pt2));
}

int main(){
    int num_p, x, y;
    ios_base::sync_with_stdio(false);
    cin.tie(0);
    while (cin >> num_p){
        if (num_p == 0)
            break;
            
        vector<point> points;
        for (int i = 0; i < num_p; i++){
            cin >> x >> y;
            points.push_back({x, y});
        }

        sort(points.begin(), points.end(), cmp);
        points.erase(unique(points.begin(), points.end(), [](point a, point b) {
                return a.x == b.x && a.y == b.y;
            }), points.end());        
        auto min_ele = min_element(points.begin(), points.end(), cmp);
        
        swap(points[0], *min_ele);
        pt_ini = points[0];
        sort(points.begin()+1, points.end(), comp_angle);
        
        int top = -1;
        for (int i = 0; i < points.size(); i++){
            while (top >= 1 && (cross(points[top-1], points[top], points[i]) <= 0))
                top--;
            points[++top] = points[i];
        }
        
        cout << top+1 << endl;
        for (int i = 0; i <= top; i++)
            cout << points[i].x << " " << points[i].y << endl;

    }
    return 0;
}