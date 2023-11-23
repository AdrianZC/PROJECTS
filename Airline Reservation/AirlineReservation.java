/*
    Name: Adrian Zhu Chou
    Email: azhuchou@ucsd.edu
    PID: A16361462
    Sources used: javatpoint, w3schools, tutors

    This file is used to read in the information from a CSV file
    and determines the amount of seat and passengers on the plane.
    This also allows the user to book, cancel and upgrade plane tickets.
    It allows for the user to search for a passenger's seat number,
    print the available ticets in each class and print a view of 
    the layout of the plane.
 */

 import java.util.Scanner;
 import java.io.FileNotFoundException;
 import java.io.File;
 public class AirlineReservation {
    /* Delimiters and Formatters */
    private static final String CSV_DELIMITER = ",";
    private static final String COMMAND_DELIMITER = " ";
    private static final String PLANE_FORMAT = "%d\t | %s | %s \n";

    /* Travel Classes */
    private static final int FIRST_CLASS = 0;
    private static final int BUSINESS_CLASS = 1;
    private static final int ECONOMY_CLASS = 2;
    private static final String[] CLASS_LIST = new String[] {"F", "B", "E"};
    private static final String[] CLASS_FULLNAME_LIST = new String[] {
        "First Class", "Business Class", "Economy Class"};

    /* Commands */
    private static final String[] COMMANDS_LIST = new String[] { "book", 
        "cancel", "lookup", "availabletickets", "upgrade", "print","exit"};
    private static final int BOOK_IDX = 0;
    private static final int CANCEL_IDX = 1;
    private static final int LOOKUP_IDX = 2;
    private static final int AVAI_TICKETS_IDX = 3;
    private static final int UPGRADE_IDX = 4;
    private static final int PRINT_IDX = 5;
    private static final int EXIT_IDX = 6;
    private static final int BOOK_UPGRADE_NUM_ARGS = 3;
    private static final int CANCEL_LOOKUP_NUM_ARGS = 2;

    /* Strings for main */
    private static final String USAGE_HELP =
            "Available commands:\n" +
            "- book <travelClass(F/B/E)> <passengerName>\n" +
            "- book <rowNumber> <passengerName>\n" +
            "- cancel <passengerName>\n" +
            "- lookup <passengerName>\n" +
            "- availabletickets\n" +
            "- upgrade <travelClass(F/B)> <passengerName>\n" +
            "- print\n" +
            "- exit";
    private static final String CMD_INDICATOR = "> ";
    private static final String INVALID_COMMAND = "Invalid command.";
    private static final String INVALID_ARGS = "Invalid number of arguments.";
    private static final String INVALID_ROW = 
        "Invalid row number %d, failed to book.\n";
    private static final String DUPLICATE_BOOK =
        "Passenger %s already has a booking and cannot book multiple seats.\n";
    private static final String BOOK_SUCCESS = 
        "Booked passenger %s successfully.\n";
    private static final String BOOK_FAIL = "Could not book passenger %s.\n";
    private static final String CANCEL_SUCCESS = 
        "Canceled passenger %s's booking successfully.\n";
    private static final String CANCEL_FAIL = 
        "Could not cancel passenger %s's booking, do they have a ticket?\n";
    private static final String UPGRADE_SUCCESS = 
        "Upgraded passenger %s to %s successfully.\n";
    private static final String UPGRADE_FAIL = 
        "Could not upgrade passenger %s to %s.\n";
    private static final String LOOKUP_SUCCESS = 
            "Passenger %s is in row %d.\n";
    private static final String LOOKUP_FAIL = "Could not find passenger %s.\n";
    private static final String AVAILABLE_TICKETS_FORMAT = "%s: %d\n";

    /* Own final variables */
    private static final int PLANE_ROWS_INDEX = 0;
    private static final int FIRST_CLASS_INDEX = 1;
    private static final int BUSINESS_CLASS_INDEX = 2;
    private static final int AVAILABLE_TICKETS_SIZE = 3;
    
    /* Static variables - DO NOT add any additional static variables */
    static String [] passengers;
    static int planeRows;
    static int firstClassRows;
    static int businessClassRows;

    /**
     * Runs the command-line interface for our Airline Reservation System.
     * Prompts user to enter commands, which correspond to different functions.
     * @param args args[0] contains the filename to the csv input
     * @throws FileNotFoundException if the filename args[0] is not found
     */
    public static void main (String[] args) throws FileNotFoundException {
        // If there are an incorrect num of args, print error message and quit
        if (args.length != 1) {
            System.out.println(INVALID_ARGS);
            return;
        }
        initPassengers(args[0]); // Populate passengers based on csv input file
        System.out.println(USAGE_HELP);
        Scanner scanner = new Scanner(System.in);
        while (true) {
            System.out.print(CMD_INDICATOR);
            String line = scanner.nextLine().trim();

            // Exit
            if (line.toLowerCase().equals(COMMANDS_LIST[EXIT_IDX])) {
                scanner.close();
                return;
            }

            String[] splitLine = line.split(COMMAND_DELIMITER);
            splitLine[0] = splitLine[0].toLowerCase(); 

            // Check for invalid commands
            boolean validFlag = false;
            for (int i = 0; i < COMMANDS_LIST.length; i++) {
                if (splitLine[0].toLowerCase().equals(COMMANDS_LIST[i])) {
                    validFlag = true;
                }
            }
            if (!validFlag) {
                System.out.println(INVALID_COMMAND);
                continue;
            }

            // Book
            if (splitLine[0].equals(COMMANDS_LIST[BOOK_IDX])) {
                if (splitLine.length < BOOK_UPGRADE_NUM_ARGS) {
                    System.out.println(INVALID_ARGS);
                    continue;
                }
                String[] contents = line.split(COMMAND_DELIMITER, 
                        BOOK_UPGRADE_NUM_ARGS);
                String passengerName = contents[contents.length - 1];
                try {
                    // book row <passengerName>
                    int row = Integer.parseInt(contents[1]);
                    if (row < 0 || row >= passengers.length) {
                        System.out.printf(INVALID_ROW, row);
                        continue;
                    }
                    // Do not allow duplicate booking
                    boolean isDuplicate = false;
                    for (int i = 0; i < passengers.length; i++) {
                        if (passengerName.equals(passengers[i])) {
                            isDuplicate = true;
                        }
                    }
                    if (isDuplicate) {
                        System.out.printf(DUPLICATE_BOOK, passengerName);
                        continue;
                    }
                    if (book(row, passengerName)) {
                        System.out.printf(BOOK_SUCCESS, passengerName);
                    } else {
                        System.out.printf(BOOK_FAIL, passengerName);
                    }
                } catch (NumberFormatException e) {
                    // book <travelClass(F/B/E)> <passengerName>
                    validFlag = false;
                    contents[1] = contents[1].toUpperCase();
                    for (int i = 0; i < CLASS_LIST.length; i++) {
                        if (CLASS_LIST[i].equals(contents[1])) {
                            validFlag = true;
                        }
                    }
                    if (!validFlag) {
                        System.out.println(INVALID_COMMAND);
                        continue;
                    }
                    // Do not allow duplicate booking
                    boolean isDuplicate = false;
                    for (int i = 0; i < passengers.length; i++) {
                        if (passengerName.equals(passengers[i])) {
                            isDuplicate = true;
                        }
                    }
                    if (isDuplicate) {
                        System.out.printf(DUPLICATE_BOOK, passengerName);
                        continue;
                    }
                    int travelClass = FIRST_CLASS;
                    if (contents[1].equals(CLASS_LIST[BUSINESS_CLASS])) {
                        travelClass = BUSINESS_CLASS;
                    } else if (contents[1].equals(
                                CLASS_LIST[ECONOMY_CLASS])) {
                        travelClass = ECONOMY_CLASS;
                    }
                    if (book(passengerName, travelClass)) {
                        System.out.printf(BOOK_SUCCESS, passengerName);
                    } else {
                        System.out.printf(BOOK_FAIL, passengerName);
                    }
                }
            }

            // Upgrade 
            if (splitLine[0].equals(COMMANDS_LIST[UPGRADE_IDX])) {
                if (splitLine.length < BOOK_UPGRADE_NUM_ARGS) {
                    System.out.println(INVALID_ARGS);
                    continue;
                }
                String[] contents = line.split(COMMAND_DELIMITER, 
                        BOOK_UPGRADE_NUM_ARGS);
                String passengerName = contents[contents.length - 1];
                validFlag = false;
                contents[1] = contents[1].toUpperCase();
                for (int i = 0; i < CLASS_LIST.length; i++) {
                    if (CLASS_LIST[i].equals(contents[1])) {
                        validFlag = true;
                    }
                }
                if (!validFlag) {
                    System.out.println(INVALID_COMMAND);
                    continue;
                }
                int travelClass = FIRST_CLASS;
                if (contents[1].equals(CLASS_LIST[BUSINESS_CLASS])) {
                    travelClass = BUSINESS_CLASS;
                } else if (contents[1].equals(CLASS_LIST[ECONOMY_CLASS])) {
                    travelClass = ECONOMY_CLASS;
                }
                if (upgrade(passengerName, travelClass)) {
                    System.out.printf(UPGRADE_SUCCESS, passengerName, 
                            CLASS_FULLNAME_LIST[travelClass]);
                } else {
                    System.out.printf(UPGRADE_FAIL, passengerName, 
                            CLASS_FULLNAME_LIST[travelClass]);
                }
            }

            // Cancel
            if (splitLine[0].equals(COMMANDS_LIST[CANCEL_IDX])) {
                if (splitLine.length < CANCEL_LOOKUP_NUM_ARGS) {
                    System.out.println(INVALID_ARGS);
                    continue;
                }
                String[] contents = line.split(COMMAND_DELIMITER, 
                        CANCEL_LOOKUP_NUM_ARGS);
                String passengerName = contents[contents.length - 1];
                if (cancel(passengerName)) {
                    System.out.printf(CANCEL_SUCCESS, passengerName);
                } else {
                    System.out.printf(CANCEL_FAIL, passengerName);
                }
            }

            // Lookup
            if (splitLine[0].equals(COMMANDS_LIST[LOOKUP_IDX])) {
                if (splitLine.length < CANCEL_LOOKUP_NUM_ARGS) {
                    System.out.println(INVALID_ARGS);
                    continue;
                }
                String[] contents = line.split(COMMAND_DELIMITER, 
                        CANCEL_LOOKUP_NUM_ARGS);
                String passengerName = contents[contents.length - 1];
                if (lookUp(passengerName) == -1) {
                    System.out.printf(LOOKUP_FAIL, passengerName);
                } else {
                    System.out.printf(LOOKUP_SUCCESS, passengerName, 
                            lookUp(passengerName));
                }
            }

            // Available tickets
            if (splitLine[0].equals(COMMANDS_LIST[AVAI_TICKETS_IDX])) {
                int[] numTickets = availableTickets();
                for (int i = 0; i < CLASS_FULLNAME_LIST.length; i++) {
                    System.out.printf(AVAILABLE_TICKETS_FORMAT, 
                            CLASS_FULLNAME_LIST[i], numTickets[i]);
                }
            }

            // Print
            if (splitLine[0].equals(COMMANDS_LIST[PRINT_IDX])) {
                printPlane();
            }
        }
    }

    /**
     * Initializes the static variables passengers, planeRows, firstClassRows,
     * and businessClassRows using the contents of the CSV file named fileName.
     * 
     * @param fileName
     * @throws FileNotFoundException
     */
    private static void initPassengers(String fileName) throws 
            FileNotFoundException {

                File file = new File(fileName);
                Scanner input = new Scanner(file);

                String line = input.nextLine();
                String[] tmp = line.split(CSV_DELIMITER);

                // initialize the static variables for planeRows, firstClassRows, and businessClassRows
                if (input.hasNext())
                {
                    planeRows = Integer.parseInt(tmp[PLANE_ROWS_INDEX]);
                    firstClassRows = Integer.parseInt(tmp[FIRST_CLASS_INDEX]);
                    businessClassRows = Integer.parseInt(tmp[BUSINESS_CLASS_INDEX]);
                }
                else
                {
                    return;
                }

                passengers = new String[planeRows];
                
                // initialize the static variable for passengers
                while (input.hasNext())
                {
                    String secondLine = input.nextLine();
                    String[] temp = secondLine.split(CSV_DELIMITER);

                    passengers[Integer.parseInt(temp[0])] = temp[FIRST_CLASS_INDEX];
                } // TODO
    }

    /**
     * This method returns the travel class of the passenger
     * in a given row.
     * 
     * @param row 
     * @return travel class corresponding to the given row
     * @return -1 if row doesn't exist
     */
    private static int findClass(int row) {
        if (row < 0)
        {
            return -1;
        }
        else if (row < firstClassRows)
        {
            return FIRST_CLASS;
        }
        else if (row < businessClassRows + firstClassRows)
        {
            return BUSINESS_CLASS;
        }
        else if (row < planeRows)
        {
            return ECONOMY_CLASS;
        }
        else
        {
            return -1;
        } //TODO
    }

    /**
     * This method returns the first row of the given
     * travel class.
     * 
     * @param travelClass
     * @return first row of travel class
     * @return -1 if travel class is not first, business or economy
     */
    private static int findFirstRow(int travelClass) {
        if (travelClass == FIRST_CLASS)
        {
            return 0;
        }
        else if (travelClass == BUSINESS_CLASS)
        {
            return firstClassRows;
        }
        else if (travelClass == ECONOMY_CLASS)
        {
            return firstClassRows + businessClassRows;
        }
        else
        {
            return -1;
        } //TODO
    }

    /**
     * This method returns the last row of the given
     * travel class
     * 
     * @param travelClass
     * @return last row of travel class
     * @return -1 if travel class is not first, business or economy
     */
    private static int findLastRow(int travelClass) {
        if (travelClass == FIRST_CLASS)
        {
            return firstClassRows - 1;
        }
        else if (travelClass == BUSINESS_CLASS)
        {
            return businessClassRows + firstClassRows - 1;
        }
        else if (travelClass == ECONOMY_CLASS)
        {
            return planeRows - 1;
        }
        else
        {
            return -1;
        } //TODO
    }

    /**
     * This method books the passenger in their travel class.
     * 
     * @param passengerName
     * @param travelClass
     * @return true if a seat is available for travel class
     * @return false if a seat isn't available for travel class 
     * or passenger name is null
     */
    public static boolean book(String passengerName, int travelClass) {

        int firstRowOfClass = findFirstRow(travelClass);
        int lastRowOfClass = findLastRow(travelClass);

        if (passengerName == null || passengerName.equals("null"))
        {
            return false;
        }
        else
        {
            for (int i = firstRowOfClass; i < lastRowOfClass + 1; i++)
            {
                if (passengers[i] == null)
                {
                    passengers[i] = passengerName;
                    return true;
                }
            }
        }

        return false; // TODO
    }

    /**
     * This method books the passenger at their seat number for their class.
     * 
     * @param row
     * @param passengerName
     * @return true if there is an available seat for their class
     * @return false if there is no available seat for their class
     * or passenger name is null
     */
    public static boolean book(int row, String passengerName) {
        if (passengerName == null || passengerName.equals("null"))
        {
            return false;
        }
        else
        {
            if (passengers[row] == null)
            {
                passengers[row] = passengerName;
                return true;
            }
            else
            {
                return book(passengerName, findClass(row));
            }
        } // TODO
    }

    /**
     * This method cancels the booking for the passenger.
     * 
     * @param passengerName
     * @return true if removal was successful
     * @return false if removal was unsuccesful or passenger name
     * is null
     */
    public static boolean cancel(String passengerName){
        if (passengerName == null || passengerName.equals("null"))
        {
            return false;
        }
        for(int i = 0; i < passengers.length; i++)
        {
            if (passengers[i] == passengerName)
            {
                passengers[i] = null;
                return true;
            }
        }

        return false; // TODO
    }

    /**
     * This method looks up the row number of the passenger
     * 
     * @param passengerName
     * @return row number of passenger
     * @return -1 if row number not found or passenger name is null
     */
    public static int lookUp(String passengerName) {
        if (passengerName == null || passengerName.equals("null"))
        {
            return -1;
        }
        for(int i = 0; i < passengers.length; i++)
        { 
            if (passengers[i] != null && passengers[i].equals(passengerName))
            {
                return i;
            }
        }
        
        return -1; // TODO
    }

    /**
     * This method shows the number of available tickes in each travel class.
     * 
     * @return an array of available tickets in each class
     */
    public static int[] availableTickets() {
        int[] availableTickets = new int[AVAILABLE_TICKETS_SIZE];
        
        availableTickets[PLANE_ROWS_INDEX] = firstClassRows;
        availableTickets[FIRST_CLASS_INDEX] = businessClassRows;
        availableTickets[BUSINESS_CLASS_INDEX] = planeRows - firstClassRows - businessClassRows;

        for(int i = 0; i < passengers.length; i++)
        {
            if (passengers[i] != null)
            {
                if (i < firstClassRows)
                {
                    availableTickets[PLANE_ROWS_INDEX]--;
                }
                else if (i < businessClassRows + firstClassRows)
                {
                    availableTickets[FIRST_CLASS_INDEX]--;
                }
                else if (i < planeRows + businessClassRows + firstClassRows)
                {
                    availableTickets[BUSINESS_CLASS_INDEX]--;
                }
            }
        }

        return availableTickets; // TODO
    }

    /**
     * This method upgrades a passengers ticket.
     * 
     * @param passengerName
     * @param upgradeClass
     * @return false if passenger name is null or not found or 
     * upgrade is lower or equal to passenger current class
     * @return true if upgrade was successful
     * @return false if upgrade wasn't successful
     */
    public static boolean upgrade(String passengerName, int upgradeClass) {
        if (passengerName == null || passengerName.equals("null"))
        {
            return false;
        }
        for (int i = 0; i < passengers.length; i++)
        {
            if (passengers[i] != null && passengers[i].equals(passengerName))
            {   
                if (upgradeClass >= findClass(i))
                {
                    return false;
                }
                else
                {
                    passengers[i] = null;
                    if (book(passengerName, upgradeClass) == false)
                    {
                        passengers[i] = passengerName;
                        return false;
                    }
                    return true;    
                }
            }
        }

        return false; // TODO
    }

    /**
     * Prints out the names of each of the passengers according to their booked
     * seat row. No name is printed for an empty (currently available) seat.
     */
    public static void printPlane() {
        for (int i = 0; i < passengers.length; i++) {
            System.out.printf(PLANE_FORMAT, i, CLASS_LIST[findClass(i)], 
                    passengers[i] == null ? "" : passengers[i]);
        }
    }
}
