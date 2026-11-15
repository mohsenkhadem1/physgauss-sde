import com.comsol.model.*;
import com.comsol.model.util.*;
import com.comsol.model.physics.*;
import java.io.*;
import java.util.*;

/* COMSOL 6.4 re-solve of the Figure-1 reference scenarios (independent check of the in-house T10 truth).
   Usage: java Fig1Comsol SCEN WORLD HMAX RTOL MAXSTEP OUTNAME [HTIP]
     SCEN  S1 (stem push D2c) | S2 (block press D2) | S3 (stem striker D3)
     WORLD W500|W1000|W2000 (S1,S3) or Bleft|Bcentre|Bright (S2)
     HMAX  max tet size [m] (quadratic Lagrange tets);  HTIP: S3 only, max size in the contact zone x>0.12
     RTOL  relative tolerance of the BDF time stepper;  MAXSTEP max time step [s]
   Writes runs/fig1-v01/comsol/OUTNAME.txt : one row per saved frame: t, then (ux,uy,uz) per station/probe,
   then (S3) cx cy cz cxt cyt czt of the striker. And OUTNAME.log.json with setup + timings. */
public class Fig1Comsol {
  static String OUT = "/home/bayespc2/vscode/guassianSDE/runs/fig1-v01/comsol/";
  static double[][] STATIONS = {{-.11, -.09}, {-.01, .01}, {.09, .11}, {.18, .20}};

  static void boxSel(ModelNode c, String tag, int dim, double[] lo, double[] hi) {
    c.selection().create(tag, "Box");
    c.selection(tag).set("entitydim", String.valueOf(dim));
    c.selection(tag).set("xmin", lo[0]); c.selection(tag).set("xmax", hi[0]);
    c.selection(tag).set("ymin", lo[1]); c.selection(tag).set("ymax", hi[1]);
    c.selection(tag).set("zmin", lo[2]); c.selection(tag).set("zmax", hi[2]);
    c.selection(tag).set("condition", "inside");
  }

  static void intop(ModelNode c, String tag, String sel) {
    c.cpl().create(tag, "Integration"); c.cpl(tag).selection().geom("g", 3);
    if (sel == null) c.cpl(tag).selection().all(); else c.cpl(tag).selection().named(sel);
    try { c.cpl(tag).set("frame", "material"); } catch (Throwable e) { System.out.println("cpl frame not settable: " + e); }
  }

  static void block(GeomSequence g, String tag, double[] lo, double[] hi) {
    g.create(tag, "Block");
    g.feature(tag).set("pos", new double[]{lo[0], lo[1], lo[2]});
    g.feature(tag).set("size", new double[]{hi[0]-lo[0], hi[1]-lo[1], hi[2]-lo[2]});
  }

  static void hyper(Physics s, String tag, String sel, String mu, String lam, String rho) {
    s.create(tag, "HyperelasticModel", 3);
    if (sel == null) s.feature(tag).selection().all(); else s.feature(tag).selection().named(sel);
    s.feature(tag).set("MaterialModel", "userDefined");
    s.feature(tag).set("Ws", "0.5*" + mu + "*(solid.Jel^(2/3)*solid.I1CIel-3)-" + mu + "*log(solid.Jel)+0.5*" + lam + "*log(solid.Jel)^2");
    s.feature(tag).set("rho_mat", "userdef"); s.feature(tag).set("rho", rho);
  }

  static void load(Physics s, String tag, String sel, String fx, String fy, String fz) {
    s.create(tag, "BodyLoad", 3);
    if (sel == null) s.feature(tag).selection().all(); else s.feature(tag).selection().named(sel);
    s.feature(tag).set("forceType", "ForceVolume");  // force per REFERENCE volume
    s.feature(tag).set("forceReferenceVolume", new String[]{fx, fy, fz});
    System.out.println("load " + tag + " = " + String.join(",", s.feature(tag).getStringArray("forceReferenceVolume")));
  }

  public static void main(String[] a) { try {
    String scen = a[0], world = a[1], name = a[5];
    double hmax = Double.parseDouble(a[2]), rtol = Double.parseDouble(a[3]), maxstep = Double.parseDouble(a[4]);
    double htip = a.length > 6 ? Double.parseDouble(a[6]) : hmax;
    long t0 = System.currentTimeMillis();
    ModelUtil.initStandalone(false);
    ModelUtil.setPreference("cluster.processor.numberofprocessors", System.getenv().getOrDefault("NPROC", "4"));
    ModelUtil.setPreference("cluster.processor.numberofprocessorsactive", "on");
    ModelUtil.showProgress(OUT + name + ".progress");
    Model m = ModelUtil.create("M");
    ModelNode c = m.component().create("c", true);
    GeomSequence g = c.geom().create("g", 3);
    boolean stem = !scen.equals("S2");
    double rho, E, nu, tend; int nframes;
    List<String> probeExpr = new ArrayList<>();
    if (stem) {
      double[] W = world.equals("W500") ? new double[]{500, 2e6} : world.equals("W1000") ? new double[]{1000, 4e6} : new double[]{2000, 8e6};
      rho = W[0]; E = W[1]; nu = 0.3; tend = 0.75; nframes = 37;
      // free stem x in [-0.20,0.20]; the clamp block x<-0.20 of the truth is fully fixed, so it is equivalent to fixing the face x=-0.20.
      double[] cuts = scen.equals("S1") ? new double[]{-.20, -.11, -.09, -.01, .01, .085, .09, .11, .115, .18, .20}
                                        : new double[]{-.20, -.11, -.09, -.01, .01, .09, .11, .12, .18, .20};
      for (int i = 0; i + 1 < cuts.length; i++) block(g, "b" + i, new double[]{cuts[i], .56, -.04}, new double[]{cuts[i+1], .64, .04});
      g.run();
      double e = 1e-4;
      for (int s = 0; s < 4; s++) { boxSel(c, "st" + s, 3, new double[]{STATIONS[s][0]-e, .5, -.1}, new double[]{STATIONS[s][1]+e, .7, .1}); intop(c, "ist" + s, "st" + s); }
      boxSel(c, "clampface", 2, new double[]{-.20-e, .5, -.1}, new double[]{-.20+e, .7, .1});
      if (scen.equals("S1")) boxSel(c, "slab", 3, new double[]{.085-e, .5, -.1}, new double[]{.115+e, .7, .1});
      else boxSel(c, "tip", 3, new double[]{.12-e, .5, -.1}, new double[]{.20+e, .7, .1});
      for (int s = 0; s < 4; s++) for (String q : new String[]{"u", "v", "w"}) probeExpr.add("ist" + s + "(" + q + ")/ist" + s + "(1)");
    } else {
      double xc = world.equals("Bleft") ? -0.06 : world.equals("Bright") ? 0.06 : 0.0;
      rho = 1000; E = 6e4; nu = 0.4; tend = 0.5; nframes = 25;
      block(g, "body", new double[]{-.12, .53, -.06}, new double[]{.12, .63, .06});
      block(g, "incl", new double[]{xc-.04, .555, -.06}, new double[]{xc+.04, .63, .06});
      block(g, "press", new double[]{-.09, .62, -.02}, new double[]{-.05, .63, .02});
      g.run();
      double e = 1e-4;
      boxSel(c, "incl", 3, new double[]{xc-.04-e, .555-e, -.06-e}, new double[]{xc+.04+e, .63+e, .06+e});
      boxSel(c, "press", 3, new double[]{-.09-e, .62-e, -.02-e}, new double[]{-.05+e, .63+e, .02+e});
      boxSel(c, "bottom", 2, new double[]{-.2, .53-e, -.1}, new double[]{.2, .53+e, .1});
      intop(c, "ipress", "press");
    }
    System.out.println("domains: " + c.geom("g").getNDomains());
    double mu = E/(2*(1+nu)), lam = E*nu/((1+nu)*(1-2*nu));
    m.param().set("mu0", String.valueOf(mu)); m.param().set("lam0", String.valueOf(lam));
    m.param().set("rho0", String.valueOf(rho));
    m.param().set("ramp", "0");
    Physics s = c.physics().create("solid", "SolidMechanics", "g");
    s.prop("ShapeProperty").set("order_displacement", name.contains("_p3") ? "3" : "2");  // quadratic Lagrange (cubic for the p-refinement check, OUTNAME containing _p3)
    System.out.println("order_displacement=" + s.prop("ShapeProperty").getString("order_displacement"));
    hyper(s, "hmm1", null, "mu0", "lam0", "rho0");
    if (!stem) {
      m.param().set("mu1", String.valueOf(8*mu)); m.param().set("lam1", String.valueOf(8*lam));
      hyper(s, "hmm2", "incl", "mu1", "lam1", "rho0");
    }
    String ramp = "0.5*(1-cos(pi*min(t/0.25[s],1)))";
    load(s, "grav", null, "0", "-rho0*9.8", "0");
    if (scen.equals("S1")) {
      String V = "(0.03*0.08*0.08)";
      load(s, "push", "slab", "0", "-16*" + ramp + "/" + V, "-12*" + ramp + "/" + V);
    } else if (scen.equals("S2")) {
      String V = "(0.04*0.01*0.04)";
      load(s, "push", "press", "0", "-30*" + ramp + "/" + V, "0");
    } else {
      // rigid striker: 3 translational ODE states, volumetric penalty on stem material inside the sphere
      m.param().set("kpen", "1e11"); m.param().set("rs", "0.03"); m.param().set("ms", "0.3");
      intop(c, "iall", null);
      c.variable().create("var1"); c.variable("var1").selection().geom("g", 3); c.variable("var1").selection().all();
      c.variable("var1").set("dxs", "X+u-cx"); c.variable("var1").set("dys", "Y+v-cy"); c.variable("var1").set("dzs", "Z+w-cz");
      c.variable("var1").set("dd", "sqrt(dxs^2+dys^2+dzs^2)");
      c.variable("var1").set("pen", "kpen*max(rs-dd,0)/dd");
      c.variable("var1").set("bxs", "pen*dxs"); c.variable("var1").set("bys", "pen*dys"); c.variable("var1").set("bzs", "pen*dzs");
      load(s, "contact", null, "bxs", "bys", "bzs");
      s.create("ge1", "GlobalEquations", -1);
      s.feature("ge1").set("name", new String[]{"cx", "cy", "cz"});
      s.feature("ge1").set("equation", new String[]{"ms*cxtt+iall(bxs)", "ms*cytt+iall(bys)", "ms*cztt+iall(bzs)"});
      s.feature("ge1").set("initialValueU", new String[]{"0.17", "0.60", "0.0755"});
      s.feature("ge1").set("initialValueUt", new String[]{"0", "0", "-1"});
      try { s.feature("ge1").set("DependentVariableQuantity", "displacement"); } catch (Throwable ex) { System.out.println("dvq: " + ex.getMessage()); }
      try { s.feature("ge1").set("SourceTermQuantity", "force"); } catch (Throwable ex) { System.out.println("stq: " + ex.getMessage()); }
    }
    s.create("fix1", "Fixed", 2); s.feature("fix1").selection().named(stem ? "clampface" : "bottom");
    // mesh
    MeshSequence me = c.mesh().create("me");
    me.create("ftet", "FreeTet");
    me.feature("size").set("custom", "on"); me.feature("size").set("hmax", hmax); me.feature("size").set("hmin", hmax/4);
    if (scen.equals("S3") && htip < hmax) {
      me.feature("ftet").create("sz", "Size"); me.feature("ftet").feature("sz").selection().geom("g", 3);
      me.feature("ftet").feature("sz").selection().named("tip");
      me.feature("ftet").feature("sz").set("custom", "on"); me.feature("ftet").feature("sz").set("hmaxactive", true); me.feature("ftet").feature("sz").set("hmax", htip);
    }
    me.run();
    int nel = me.getNumElem("tet");
    System.out.println("tets: " + nel); System.out.flush();
    // study: time dependent, BDF with error control
    String tlist = "range(0,1/48," + tend + ")";
    m.study().create("std"); m.study("std").create("time", "Transient");
    m.study("std").feature("time").set("tlist", tlist);
    m.study("std").feature("time").set("usertol", "on"); m.study("std").feature("time").set("rtol", String.valueOf(rtol));
    m.study("std").feature("time").set("geometricNonlinearity", "on");
    m.sol().create("sol1"); m.sol("sol1").study("std"); m.sol("sol1").createAutoSequence("std");
    SolverFeature t1 = m.sol("sol1").feature("t1");
    t1.set("timemethod", "bdf"); t1.set("tstepsbdf", "free"); t1.set("rtol", String.valueOf(rtol));
    t1.set("maxstepconstraintbdf", "const"); t1.set("maxstepbdf", String.valueOf(maxstep));
    t1.set("initialstepbdfactive", "on"); t1.set("initialstepbdf", String.valueOf(maxstep/10));
    t1.set("maxorder", "2");
    t1.set("tout", "tlist");
    System.out.println("t1: method=" + t1.getString("timemethod") + " rtol=" + t1.getString("rtol") + " maxstep=" + t1.getString("maxstepbdf")
        + " maxorder=" + t1.getString("maxorder") + " atolglobal=" + t1.getString("atolglobal") + " atolmethod=" + t1.getString("atolglobalvaluemethod"));
    for (String ft : t1.feature().tags()) System.out.println("  t1 sub " + ft);
    long tb = System.currentTimeMillis();
    m.sol("sol1").runAll();
    long ts = System.currentTimeMillis();
    System.out.println("solved in " + (ts-tb)/1000.0 + " s"); System.out.flush();
    // evaluation
    List<String> ex = new ArrayList<>(); ex.add("t");
    if (stem) ex.addAll(probeExpr);
    if (scen.equals("S3")) for (String q : new String[]{"cx", "cy", "cz", "cxt", "cyt", "czt"}) ex.add(q);
    if (scen.equals("S2")) ex.add("ipress(1)");
    m.result().numerical().create("eg", "EvalGlobal");
    m.result().numerical("eg").set("data", "dset1");
    m.result().numerical("eg").set("expr", ex.toArray(new String[0]));
    double[][] G = m.result().numerical("eg").getReal();  // [expr][time]
    int nt = G[0].length;
    {  // frame check: integration volumes at the last frame must equal the REFERENCE volumes
      String[] vx = stem ? (scen.equals("S3") ? new String[]{"ist0(1)", "iall(1)"} : new String[]{"ist0(1)"}) : new String[]{"ipress(1)"};
      m.result().numerical().create("eg2", "EvalGlobal"); m.result().numerical("eg2").set("data", "dset1");
      m.result().numerical("eg2").set("expr", vx); double[][] V = m.result().numerical("eg2").getReal();
      for (int i = 0; i < vx.length; i++) System.out.println("volume check " + vx[i] + " at t_end = " + V[i][V[i].length-1] + " (frame " + c.cpl(vx[i].substring(0, vx[i].indexOf('('))).getString("frame") + ")");
    }
    double[][][] P = null;
    if (!stem) {
      double[] px = new double[9]; for (int i = 0; i < 9; i++) px[i] = -0.10 + 0.025*i;
      double[][] coord = new double[3][27]; int k = 0;
      for (double z : new double[]{-.04, 0, .04}) for (double x : px) { coord[0][k] = x; coord[1][k] = .63; coord[2][k] = z; k++; }
      m.result().numerical().create("ip", "Interp");
      m.result().numerical("ip").set("data", "dset1");
      m.result().numerical("ip").set("expr", new String[]{"u", "v", "w"});
      m.result().numerical("ip").setInterpolationCoordinates(coord);
      P = m.result().numerical("ip").getData();  // [expr][solnum][coord]
      System.out.println("press volume (ipress(1)) = " + G[G.length-1][0]);
    }
    try (PrintWriter w = new PrintWriter(new FileWriter(OUT + name + ".txt"))) {
      for (int j = 0; j < nt; j++) {
        StringBuilder sb = new StringBuilder(); sb.append(String.format("%.10g", G[0][j]));
        if (stem) {
          for (int i = 1; i <= 12; i++) sb.append(" ").append(String.format("%.10g", G[i][j]));
          if (scen.equals("S3")) for (int i = 13; i < 19; i++) sb.append(" ").append(String.format("%.10g", G[i][j]));
        } else {
          for (int p = 0; p < 27; p++) for (int q = 0; q < 3; q++) sb.append(" ").append(String.format("%.10g", P[q][j][p]));
        }
        w.println(sb);
      }
    }
    String info = String.format("{\"scen\":\"%s\",\"world\":\"%s\",\"hmax\":%g,\"htip\":%g,\"rtol\":%g,\"maxstep\":%g,\"tets\":%d,\"frames\":%d,\"solve_s\":%.1f,\"total_s\":%.1f,\"order\":\"%s\",\"timemethod\":\"bdf maxorder 2\"}",
        scen, world, hmax, htip, rtol, maxstep, nel, nt, (ts-tb)/1000.0, (System.currentTimeMillis()-t0)/1000.0, s.prop("ShapeProperty").getString("order_displacement"));
    try (PrintWriter w = new PrintWriter(new FileWriter(OUT + name + ".json"))) { w.println(info); }
    System.out.println(info);
    ModelUtil.disconnect(); System.exit(0);
  } catch (Throwable e) { e.printStackTrace(); System.exit(1); } }
}
