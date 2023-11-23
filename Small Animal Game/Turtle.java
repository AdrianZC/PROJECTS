import java.awt.Color;
import java.util.Random;

public class Turtle extends Critter {
    private static final String SPECIES_NAME = "Tu";

    public Turtle(){
        super(SPECIES_NAME);
    }

    /**
     * Returns the color of the Turtle
     * 
     * @return Color green
     */
    @Override 
    public Color getColor() {
        return Color.GREEN;
    }

    @Override
    public Direction getMove() {
        return Direction.WEST;
    }
    
    @Override
    public boolean eat() {
        for (Direction dir : Direction.values()) {
            if (info.getNeighbor(dir) != null &&
            info.getNeighbor(dir).equals("Tu")) {
                return false;
            }
        }
        return true;
    }

    @Override
    public Attack getAttack(String opponent){

       Random rand = new Random();
       double number = rand.nextDouble();

       if (number < 0.5) {
           return Attack.ROAR;
       }
       else {
           return Attack.FORFEIT;
       }
    }
}
