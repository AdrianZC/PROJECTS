import java.awt.Color;

public class Lion extends Feline {
    private int moveCount;
    private int winCount;
    private boolean sleeping;
    private boolean hungry;


    public Lion() {
        super();
        moveCount = 0;
        winCount = 0;
        sleeping = false;
        hungry = false;
        this.displayName = "Lion";
    }

    /**
     * Returns the color of the Lion
     * 
     * @return Color yellow
     */
    @Override 
    public Color getColor() {
        return Color.YELLOW;
    }

    @Override
    public Direction getMove() {
        moveCount++;
        if (moveCount <= 5) {
            return Direction.EAST;
        }
        else if (moveCount <= 10) {
            return Direction.SOUTH;
        }
        else if (moveCount <= 15) {
            return Direction.WEST;
        }
        else if (moveCount <= 20) {
            return Direction.NORTH;
        }
        else {
            moveCount = 0;
            return Direction.CENTER;
        }
    }

    @Override
    public boolean eat() {
        if (hungry) {
            hungry = false;
            return true;
        }
        return false;
    }

    @Override
    public void sleep() {
        sleeping = true;
        winCount = 0;
        this.displayName = "noiL";
    }

    @Override
    public void wakeup() {
        sleeping = false;
        this.displayName = "Lion";
    }

    @Override
    public void win() {
        winCount++;
        hungry = true;
    }
}
