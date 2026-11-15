import com.comsol.model.*;
import com.comsol.model.util.*;
public class LicTest {
  public static void main(String[] a) { try {
    long t0=System.currentTimeMillis(); System.out.println("start"); System.out.flush();
    ModelUtil.initStandalone(false); System.out.println("init "+(System.currentTimeMillis()-t0)); System.out.flush();
    ModelUtil.setPreference("cluster.processor.numberofprocessors","4"); ModelUtil.setPreference("cluster.processor.numberofprocessorsactive","on");
    Model m = ModelUtil.create("M");
    m.component().create("c", true); m.component("c").geom().create("g", 3);
    m.component("c").geom("g").create("b", "Block"); m.component("c").geom("g").feature("b").set("size", new String[]{"0.2","0.04","0.04"});
    m.component("c").geom("g").run();
    m.component("c").physics().create("solid", "SolidMechanics", "g");
    m.component("c").physics("solid").create("hmm1", "HyperelasticModel", 3); m.component("c").physics("solid").feature("hmm1").selection().all();
    m.component("c").physics("solid").feature("hmm1").set("MaterialModel", "userDefined");
    m.component("c").physics("solid").feature("hmm1").set("Ws", "0.5*mu0*(solid.Jel^(2/3)*solid.I1CIel-3)-mu0*log(solid.Jel)+0.5*lam0*log(solid.Jel)^2");
    m.component("c").physics("solid").feature("hmm1").set("rho_mat", "userdef"); m.component("c").physics("solid").feature("hmm1").set("rho", "1000"); System.out.println("compress options: "+m.component("c").physics("solid").feature("hmm1").getString("Compressibility_UserDef"));
    m.param().set("mu0", "1e6/(2*1.3)"); m.param().set("lam0", "1e6*0.3/(1.3*0.4)");
    m.component("c").physics("solid").create("fix1", "Fixed", 2); m.component("c").physics("solid").feature("fix1").selection().set(1);
    m.component("c").physics("solid").create("bl1", "BodyLoad", 3); m.component("c").physics("solid").feature("bl1").selection().all();
    m.component("c").physics("solid").feature("bl1").set("FperVol", new String[]{"0","-1000*9.8","0"});
    m.component("c").mesh().create("me"); m.component("c").mesh("me").automatic(true); m.component("c").mesh("me").run();
    m.study().create("s"); m.study("s").create("t", "Transient"); m.study("s").feature("t").set("tlist", "range(0,0.01,0.05)");
    System.out.println("built, solving "+(System.currentTimeMillis()-t0)); System.out.flush();
    m.study("s").run();
    System.out.println("solved OK; physics features: " + String.join(",", m.component("c").physics("solid").feature().tags()));
    System.out.println("mesh elements: " + m.component("c").mesh("me").getNumElem());
    ModelUtil.disconnect(); System.exit(0);
  } catch (Throwable e) { e.printStackTrace(); System.exit(1); } }
}
