/*
    Name: Adrian Zhu Chou
    Email: azhuchou@ucsd.edu
    PID: A16361462
    Sources used: javatpoint, w3schools, tutors

    This file is used to analyze the password that the user inputs.
    If the password meets all the requirements for a strong password
    then it tells the user the password is strong.
    If the password is missing any of the requirements
    then it suggests a stronger password by changing one of the
    characters or by adding more characters.
 */

import java.util.Scanner;

/**
 * This class stores an inputted string and is able to determine
 * if the input meets the requirements to be considered very weak,
 * weak, medium, or strong.
 * 
 * Instance variables:
 * password - the input from the user that is modified or returned
 */

public class PasswordSecurity
{
    /**
     * Constants
     */

    // Labels for commonly used variables and numbers to prevent magic numbers.
    private static final String PW_PROMPT = "Please enter a password: ";
    private static final String PW_TOO_SHORT = "Password is too short";
    private static final String PW_VERY_WEAK = "Password strength: very weak";
    private static final String PW_WEAK = "Password strength: weak";
    private static final String PW_MEDIUM = "Password strength: medium";
    private static final String PW_STRONG = "Password strength: strong";
    private static final String SUGGESTION_PROMPT = 
            "Here is a suggested stronger password: ";
    private static final String LETTER_RULE_SUGGESTION = "Cse";
    private static final String SYMBOL_RULE_SUGGESTION = "@!";

    private static final int MIN_PW_LENGTH = 8;
    private static final int VERY_WEAK_THRESHOLD = 1;
    private static final int WEAK_THRESHOLD = 2;
    private static final int MEDIUM_THRESHOLD = 3;
    private static final int STRONG_THRESHOLD = 4;
    private static final int LETTER_COUNT_THRESHOLD = 2;
    private static final int DIGIT_INTERVAL = 4;
    private static final int MOD_FACTOR = 10;

    /**
     * The constructor reads the input and determines 
     * what has to be done to the input to make it better
     * or return it as is.
     */

    public static void main (String[] args)
    {
        /*
         * This part of code asks the user to input a password.
         */
        Scanner input = new Scanner(System.in);

        System.out.print(PW_PROMPT);
        String password = input.nextLine();
        int k = password.length() % MOD_FACTOR;

        /*
         * This part of code determines if the password
         * is too short and if it's too short than 
         * it tells the user.
         * 
         * @return Password is too short
         */
        
        if (password.length() < MIN_PW_LENGTH)
        {
            System.out.println(PW_TOO_SHORT);
            return;
        }

        /*
         * This code stores the amounts of upper case, lower case, 
         * numbers and symbols from the password by iterating through
         * every character in the password.
         */

        int amountOfUpperCase = 0;
        int amountOfLowerCase = 0;
        int amountOfNumbers = 0;
        int amountOfSymbols = 0;

        for (int i = 0; i < password.length(); i++)
        {
            char character = password.charAt(i);

            if (Character.isUpperCase(character))
            {
                amountOfUpperCase++;
            }
            else if (Character.isLowerCase(character))
            {
                amountOfLowerCase++;
            }
            else if (Character.isDigit(character))
            {
                amountOfNumbers++;
            }
            else
            {
                amountOfSymbols++;
            }
        }

        /*
         * This determines the strength of the password
         * by adding to the stored passwordStrength value
         * if there is a certain character in the password
         * and then determining the strength of the password 
         * by the value of passwordStrength.
         * 
         * @return Password strength: very weak
         * @return Password strength: weak
         * @return Password strength: medium
         * @return Password strength: strong
         */

        int passwordStrength = 0;

        if (amountOfUpperCase > 0)
        {
            passwordStrength++;
        }
        if (amountOfLowerCase > 0)
        {
            passwordStrength++;
        }
        if (amountOfNumbers > 0)
        {
            passwordStrength++;
        }
        if (amountOfSymbols > 0)
        {
            passwordStrength++;
        }

        if (passwordStrength == VERY_WEAK_THRESHOLD)
        {
            System.out.println(PW_VERY_WEAK);
        }
        else if (passwordStrength == WEAK_THRESHOLD)
        {
            System.out.println(PW_WEAK);
        }
        else if (passwordStrength == MEDIUM_THRESHOLD)
        {
            System.out.println(PW_MEDIUM);
        }
        else if (passwordStrength == STRONG_THRESHOLD)
        {
            System.out.println(PW_STRONG);
            return;
        }

        /*
         * This code suggests the user a stronger version of their
         * password based on the requirements that they are missing.
         * 
         * @return Here is a suggested stronger password: Csepassword@!
         * @return Here is a suggested stronger password: pASSWORD
         * @return Here is a suggested stronger password: passWord
         * @return Here is a suggested stronger password: pASSkWORDk???
         */

        String passwordSuggestion = password;

        // Determines if the password requires prepend of Cse (Rules 1)
        if ((amountOfLowerCase + amountOfUpperCase) < LETTER_COUNT_THRESHOLD)
        {
            passwordSuggestion = LETTER_RULE_SUGGESTION + password;
        }

        // Determines if the first character of the password needs to be lowercase (Rule 2)
        else if (amountOfLowerCase == 0)
        {
            for (int i = 0; i < password.length(); i++)
            {
                char character = password.charAt(i);

                if (Character.isUpperCase(character))
                {
                    String beginningOfPassword = password.substring(0, i);
                    String lowercaseFirstCharacter = Character.toString(character).toLowerCase();
                    String endOfPassword = password.substring(i + 1);
                    passwordSuggestion = beginningOfPassword + lowercaseFirstCharacter + endOfPassword;
                    break;
                }
            }
        }

        // Determines if the highest value lower case character has to capitalized (Rule 3)
        else if (amountOfUpperCase == 0)
        {
            int index = 0;
            char highestLowerCase = 'a';

            for (int i = 0; i < password.length(); i++)
            {
                char character = password.charAt(i);

                if (character >= highestLowerCase)
                {
                    highestLowerCase = character;
                    index = i;
                }
            }

            String passwordBeginning = password.substring(0, index);
            String passwordEnd = password.substring(index + 1);
            highestLowerCase = Character.toUpperCase(highestLowerCase);
            passwordSuggestion = passwordBeginning + highestLowerCase + passwordEnd;
        }
        
        // Determines if the password needs a value added to it every 4 characters (Rule 4)
        if (amountOfNumbers == 0)
        {
            for (int i = DIGIT_INTERVAL; i <= passwordSuggestion.length(); i += (DIGIT_INTERVAL + 1))
            {
                passwordSuggestion = passwordSuggestion.substring(0, i) + k + passwordSuggestion.substring(i);
            }
        }

        // Determines if the password requires append @! (Rules 5)
        if (amountOfSymbols == 0)
        {
            passwordSuggestion = passwordSuggestion + SYMBOL_RULE_SUGGESTION;
        }

        System.out.println(SUGGESTION_PROMPT + passwordSuggestion);

   }
}