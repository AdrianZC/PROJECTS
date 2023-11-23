import java.awt.Color;
import java.util.Random;

public class Leopard extends Feline {
    protected static int confidence = 0;
    private static final int AMOUNT_DIRECTIONS  = 4;

    public Leopard() {
        super();
        this.displayName = "Lpd";
    }

    @Override
    public Color getColor() {
        return Color.RED;
    }

    @Override
    public Direction getMove() {
        Direction currDir = Direction.CENTER;
        for (Direction dir : Direction.values()) {
            if (info.getNeighbor(dir) != null && info.getNeighbor(dir).equals("Patrick")) {
                currDir = dir;
            }
            else if (info.getNeighbor(dir) == null || !info.getNeighbor(dir).equals("Patrick")) {

                Random rand = new Random();
                int num = rand.nextInt(AMOUNT_DIRECTIONS);

                if (num == 0) {
                    currDir = Direction.NORTH;
                }
                else if (num == 1) {
                    currDir = Direction.SOUTH;
                }
                else if (num == 2) {
                    currDir = Direction.EAST;
                }
                else {
                    currDir = Direction.WEST;
                }
            }
        }
        return currDir;
    }

    @Override
    public boolean eat() {
        Random rand = new Random();
        return rand.nextInt(100) < (confidence * 10);
    }

    @Override
    public void win() {
        if (confidence < 10) {
            confidence++;
        }
    }

    @Override
    public void lose() {
        if (confidence > 0) {
            confidence--;
        }
    }

    @Override
    public Attack getAttack(String opponent) {
        if (opponent.equals("Tu") || confidence > 5) {
            return Attack.POUNCE;
        }
        else {
            return generateAttack(opponent);
        }
    }

    protected Attack generateAttack(String opponent) {
        Attack currAttack = Attack.FORFEIT;

        if (opponent.equals("Patrick")) {
            return Attack.FORFEIT;
        }
        else {

            Random rand = new Random();
            int index = rand.nextInt(3);
            
            if (index == 0) {
                currAttack =  Attack.POUNCE;
            } else if (index == 1) {
                currAttack =  Attack.SCRATCH;
            } else if (index == 2) {
                currAttack = Attack.ROAR;
            }
        }
        return currAttack;
    }

    @Override
    public void reset() {
        confidence = 0;
    }
}
