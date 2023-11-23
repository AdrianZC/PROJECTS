import java.util.Random;

public class Feline extends Critter {
    private int moveCount; //counter for getMove method before random direction
    private int eatCount; //counter for eating
    private Direction currDir; //current direction
    private static final String SPECIES_NAME = "Fe";
    private static final int AMOUNT_DIRECTIONS  = 4;


    public Feline() {
        super(SPECIES_NAME);
        moveCount = 0;
        eatCount = 0;
        currDir = Direction.CENTER;
    }

    @Override
    public Direction getMove() {
        moveCount++;
        if (moveCount % 5 == 0) {

            Random rand = new Random();
            int num = rand.nextInt(AMOUNT_DIRECTIONS);

            if (num == 0) {
                currDir = Direction.NORTH;
                return Direction.NORTH;
            }
            else if (num == 1) {
                currDir = Direction.SOUTH;
                return Direction.SOUTH;
            }
            else if (num == 2) {
                currDir = Direction.EAST;
                return Direction.EAST;
            }
            else {
                currDir = Direction.WEST;
                return Direction.WEST;
            }
        }
        return currDir;
    }    

    @Override
    public boolean eat() {
        eatCount++;
        if (eatCount % 3 == 0) {
            return true;
        }
        return false;
    }

    @Override
    public Attack getAttack(String opponent){
        return Attack.POUNCE;
    }
}