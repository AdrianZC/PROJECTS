import java.awt.Color;
import java.util.Random;

public class Elephant extends Critter {
    protected static int goalX;
    protected static int goalY;

    public Elephant() {
        super("El");
        goalX = 0;
        goalY = 0;
    }

    @Override
    public Color getColor() {
        return Color.GRAY;
    }

    @Override
    public Direction getMove() {
        if (info.getX() == goalX && info.getY() == goalY) {
            Random rand = new Random();
            goalX = rand.nextInt(60);
            goalY = rand.nextInt(50);
        }
        Direction currDir = Direction.CENTER;
        if (info.getX() < goalX) {
            currDir = Direction.EAST;
        }
        else if (info.getX() > goalX) {
            currDir = Direction.WEST;
        }
        else if (info.getY() < goalY) {
            currDir = Direction.SOUTH;
        }
        else if (info.getY() > goalY) {
            currDir = Direction.NORTH;
        }
        return currDir;
    }

    @Override
    public boolean eat() {
        return true;
    }

    @Override
    public void mate() {
        incrementLevel(2);
    }

    @Override
    public void reset() {
        goalX = 0;
        goalY = 0;
    }
}
