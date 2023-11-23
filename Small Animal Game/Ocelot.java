import java.awt.Color;
import java.util.Random;

public class Ocelot extends Leopard {

    public Ocelot() {
        super();
        this.displayName = "Oce";
    }

    @Override
    public Color getColor() {
        return Color.LIGHT_GRAY;
    }

    @Override
    protected Attack generateAttack(String opponent) {
        if (opponent.equals("Lion") || opponent.equals("Fe")
        || opponent.equals("Lpd")) {
            return Attack.SCRATCH;
        }
        else {
            return Attack.POUNCE;
        }
    }
}
