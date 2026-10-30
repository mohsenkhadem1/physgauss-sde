// Exact clipped P1 boundary normal contact; refined degree-five friction rule.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
struct V {double a[3];};
struct Poly {int n;V p[32];};
static V mix(V a,V b,double t){V r;for(int k=0;k<3;k++)r.a[k]=a.a[k]*(1-t)+b.a[k]*t;return r;}
static double dot(V b,const double*c){return b.a[0]*c[0]+b.a[1]*c[1]+b.a[2]*c[2];}
static void cross(const double*a,const double*b,double*c){c[0]=a[1]*b[2]-a[2]*b[1];c[1]=a[2]*b[0]-a[0]*b[2];c[2]=a[0]*b[1]-a[1]*b[0];}
static bool clip(Poly &p,const double*c){
 if(!p.n)return true;Poly o;o.n=0;V prev=p.p[p.n-1];double vp=dot(prev,c);bool pin=vp>=0;
 for(int i=0;i<p.n;i++){V cur=p.p[i];double vc=dot(cur,c);bool cin=vc>=0;
  if(cin!=pin){double den=vp-vc;if(std::abs(den)>1e-300){if(o.n>=32)return false;o.p[o.n++]=mix(prev,cur,std::clamp(vp/den,0.,1.));}}
  if(cin){if(o.n>=32)return false;o.p[o.n++]=cur;}prev=cur;vp=vc;pin=cin;
 }p=o;return true;
}
static const double Q[7][4]={
 {.3333333333333333,.3333333333333333,.3333333333333333,.225},
 {.059715871789770,.470142064105115,.470142064105115,.132394152788506},
 {.470142064105115,.059715871789770,.470142064105115,.132394152788506},
 {.470142064105115,.470142064105115,.059715871789770,.132394152788506},
 {.797426985353087,.101286507323456,.101286507323456,.125939180544827},
 {.101286507323456,.797426985353087,.101286507323456,.125939180544827},
 {.101286507323456,.101286507323456,.797426985353087,.125939180544827}};
struct Context {const int64_t*ids;double x[3][3],v[3][3],normal[3],pressure[3];const double*center,*vel,*omega;double mu,v0;double*force,*reaction,*torque,*stats;};
static void friction_triangle(Context &c,V A,V B,V C,double area,int level){
 if(level){V AB=mix(A,B,.5),BC=mix(B,C,.5),CA=mix(C,A,.5);double a=area*.25;
  friction_triangle(c,A,AB,CA,a,level-1);friction_triangle(c,AB,B,BC,a,level-1);friction_triangle(c,CA,BC,C,a,level-1);friction_triangle(c,AB,BC,CA,a,level-1);return;}
 for(int qi=0;qi<7;qi++){V b;for(int k=0;k<3;k++)b.a[k]=Q[qi][0]*A.a[k]+Q[qi][1]*B.a[k]+Q[qi][2]*C.a[k];
  double fn=std::max(0.,dot(b,c.pressure));double point[3]={},velocity[3]={},arm[3],spin[3],relative[3],tangent[3],force[3];
  for(int j=0;j<3;j++)for(int k=0;k<3;k++){point[k]+=b.a[j]*c.x[j][k];velocity[k]+=b.a[j]*c.v[j][k];}
  for(int k=0;k<3;k++)arm[k]=point[k]-c.center[k];cross(c.omega,arm,spin);double vn=0;
  for(int k=0;k<3;k++){relative[k]=velocity[k]-c.vel[k]-spin[k];vn+=relative[k]*c.normal[k];}
  double speed2=0;for(int k=0;k<3;k++){tangent[k]=relative[k]-vn*c.normal[k];speed2+=tangent[k]*tangent[k];}double speed=std::sqrt(speed2),weight=area*Q[qi][3];
  for(int k=0;k<3;k++){force[k]=-weight*c.mu*fn*tangent[k]/std::max(speed,c.v0);c.reaction[k]-=force[k];c.stats[1]+=force[k]*relative[k];for(int j=0;j<3;j++)c.force[3*c.ids[j]+k]+=b.a[j]*force[k];}
  double tq[3];cross(arm,force,tq);for(int k=0;k<3;k++)c.torque[k]-=tq[k];
  c.stats[2]+=1;if(c.mu>0&&fn>0){c.stats[3]+=1;c.stats[6]+=weight;if(speed>=c.v0){c.stats[4]+=1;c.stats[7]+=weight;}}
 }
}
static void integrate_poly(Context &c,const Poly &p,double rest_area,int level){
 for(int i=1;i+1<p.n;i++){V A=p.p[0],B=p.p[i],C=p.p[i+1];
  double determinant=std::abs((B.a[1]-A.a[1])*(C.a[2]-A.a[2])-(B.a[2]-A.a[2])*(C.a[1]-A.a[1]));double area=rest_area*determinant;
  if(area<1e-22)continue;c.stats[5]+=area;
  double fn[3]={dot(A,c.pressure),dot(B,c.pressure),dot(C,c.pressure)};double sum=fn[0]+fn[1]+fn[2];V nodes[3]={A,B,C};
  for(int j=0;j<3;j++){double nj=0,weighted=0;for(int k=0;k<3;k++){nj+=nodes[k].a[j];weighted+=nodes[k].a[j]*fn[k];}double coeff=area*(nj*sum+weighted)/12;
   double force[3],arm[3],tq[3];for(int k=0;k<3;k++){force[k]=coeff*c.normal[k];c.force[3*c.ids[j]+k]+=force[k];c.reaction[k]-=force[k];arm[k]=c.x[j][k]-c.center[k];}cross(arm,force,tq);for(int k=0;k<3;k++)c.torque[k]-=tq[k];}
  if(c.mu>0)friction_triangle(c,A,B,C,area,level);else c.stats[2]+=1;
 }
}
extern "C" int integrate_contact(int nodes,int triangles,const int64_t*ids,const double*rest_area,const double*x,const double*v,int kind,const double*center,const double*rotation,const double*half,const double*plane_normal,const double*vel,const double*omega,double kd,double cd,double mu,double v0,int level,double*force,double*reaction,double*torque,double*stats){
 if(kind<0||kind>1||level<0||level>4||kd<=0||cd<0||v0<=0)return 1;
 std::memset(force,0,3*nodes*sizeof(double));std::memset(reaction,0,3*sizeof(double));std::memset(torque,0,3*sizeof(double));std::memset(stats,0,9*sizeof(double));
 for(int ti=0;ti<triangles;ti++){
  Context c;c.ids=ids+3*ti;c.center=center;c.vel=vel;c.omega=omega;c.mu=mu;c.v0=v0;c.force=force;c.reaction=reaction;c.torque=torque;c.stats=stats;
  for(int j=0;j<3;j++)for(int k=0;k<3;k++){c.x[j][k]=x[3*c.ids[j]+k];c.v[j][k]=v[3*c.ids[j]+k];}
  Poly inside;inside.n=3;inside.p[0]={{1,0,0}};inside.p[1]={{0,1,0}};inside.p[2]={{0,0,1}};
  double q[6][3],normals[6][3];int faces=kind==0?1:6;
  for(int f=0;f<faces;f++){
   for(int k=0;k<3;k++)normals[f][k]=kind==0?plane_normal[k]:((f%2)?1.:-1.)*rotation[3*k+f/2];
   for(int j=0;j<3;j++){q[f][j]=kind==0?0:-half[f/2];for(int k=0;k<3;k++)q[f][j]+=(c.x[j][k]-center[k])*normals[f][k];}
   double allow[3];for(int j=0;j<3;j++)allow[j]=-q[f][j];if(!clip(inside,allow))return 2;
   if(!inside.n)break;
  }
  if(!inside.n)continue;
  for(int f=0;f<faces;f++){
   Poly region=inside;bool skip=false;
   for(int g=0;g<faces&&region.n;g++)if(g!=f){double nearest[3];for(int j=0;j<3;j++)nearest[j]=q[f][j]-q[g][j];
    bool tie=true;for(int k=0;k<region.n;k++)if(std::abs(dot(region.p[k],nearest))>1e-13)tie=false;
    if(tie){if(g<f){skip=true;break;}else continue;}
    if(!clip(region,nearest))return 2;
   }
   if(skip||!region.n)continue;double depth=0;for(int k=0;k<region.n;k++)depth=std::max(depth,-dot(region.p[k],q[f]));stats[0]=std::max(stats[0],depth);
   if(depth<=1e-14)continue;
   // Exact degree-two elastic energy on each nearest-face region, before
   // removing regions whose dashpot would create tensile normal traction.
   for(int j=1;j+1<region.n;j++){V A=region.p[0],B=region.p[j],C=region.p[j+1];
    double determinant=std::abs((B.a[1]-A.a[1])*(C.a[2]-A.a[2])-(B.a[2]-A.a[2])*(C.a[1]-A.a[1]));
    double a=-dot(A,q[f]),b=-dot(B,q[f]),c=-dot(C,q[f]);
    stats[8]+=.5*kd*rest_area[ti]*determinant*(a*a+b*b+c*c+a*b+a*c+b*c)/6;
   }
   for(int k=0;k<3;k++)c.normal[k]=normals[f][k];
   for(int j=0;j<3;j++){double arm[3],spin[3],vn=0;for(int k=0;k<3;k++)arm[k]=c.x[j][k]-center[k];cross(omega,arm,spin);for(int k=0;k<3;k++)vn+=(c.v[j][k]-vel[k]-spin[k])*c.normal[k];c.pressure[j]=-kd*q[f][j]-cd*vn;}
   if(!clip(region,c.pressure))return 2;if(region.n)integrate_poly(c,region,rest_area[ti],level);
  }
 }
 return 0;
}
