/*
    Name: Adrian Zhu Chou
    Email: azhuchou@ucsd.edu
    PID: A16361462
    Sources used: javatpoint, w3schools, tutors

    This file is used to add or multiply two numbers given by the user.
    The calculation would align the decimal numbers and continue through the 
    process of adding or multiplying. If the input has no decimal, a decimal
    number would be added in the corresponding place.
 */

 /**
  * This class takes in two numbers as strings and aligns them
  * in order to perform the proper calculations and return
  * an accurate answer of the two numbers being added or 
  * multiplied.
  */
public class Calculator
{
    // Call the methods
    // public static void main (String[] args)
    // {
    //     System.out.print(Calculator.add("538.1874863896267", "539.4463845499305"));
    // }

    // Number Extraction

    /**
     * This method returns the whole number value of the number.
     * s
     * @param number user input
     * 
     * @return user input number
     * @return empty string
     * @return whole number
     */

    public static String extractWholeNumber(String number)
    {
        if (number.indexOf(".") == -1)
        {
            return number;
        }
        else if (number.substring(0, number.indexOf(".")).equals(""))
        {
            return "";
        }
        else
        { 
            return number.substring(0, number.indexOf("."));
        }  
    }
    
    /**
     * This method returns the decimal value of the number.
     * 
     * @param number user input
     * 
     * @return empty string 
     * @return decimal numbers
     */

    public static String extractDecimal(String number)
    {
        if (number.indexOf("." ) == -1)
        {
            return "";
        }
        else if (number.substring(number.indexOf(".") + 1).equals(""))
        {
            return "";
        }
        else 
        {
            return number.substring(number.indexOf(".") + 1);
        }
    }

    // Alignment and Formatting

    /**
     * This method adds the necessary amount of zeros to the beginning
     * of the number to align it with the second number.
     * 
     * @param number   user input
     * @param numZeros number of zeros to prepend
     * 
     * @return user input
     * @return user input number with prepended zeros
     */

    public static String prependZeros(String number, int numZeros)
    {
        String amountOfZeros = "";

        if (numZeros < 0)
        {
            return number;
        }

        for (int i = 0; i < numZeros; i++)
        {
            amountOfZeros += "0";
        }

        return amountOfZeros.concat(number);
    }

    /**
     * This method adds the necessary amount of zeros to the end of
     * the number to align it with the second number.
     * 
     * @param number   user input
     * @param numZeros number of zeros to append
     * 
     * @return user input
     * @return user input number with appended zeros
     */

    public static String appendZeros(String number, int numZeros)
    {
        String amountOfZeros = "";

        if (numZeros < 0)
        {
            return number;
        }
        
        for (int i = 0; i < numZeros; i++)
        {
            amountOfZeros += "0";
        }

        return number.concat(amountOfZeros);
    }

    /**
     * This method returns a number without leading or trailing zeros.
     * If the number number inputted doesn't have a decimal point, 
     * this method will add one.
     * 
     * @param number user input
     * 
     * @return number after removing leading and trailing zeros
     */

     public static String formatResult(String number) 
     {
        boolean hasDecimal = number.contains(".");

        // if there are no decimals after whole number
        if (hasDecimal == false)
        {
            number += ".";
        }

        String wholeNumber = extractWholeNumber(number);
        String decimalNumber = extractDecimal(number);
        int wholeNumberLength = wholeNumber.length();

        // remove leading zeroes
        for (int i = 0; i < wholeNumberLength && wholeNumber.charAt(0) == '0'; i++)
        {
            wholeNumber = wholeNumber.substring(1);
        }
        if (wholeNumber == "")
        {
            wholeNumber = "0";
        }

        //remove trailing zeroes
        for (int j = decimalNumber.length() - 1; j >= 0 && decimalNumber.charAt(j) == '0'; j--)
        {
            decimalNumber = decimalNumber.substring(0, j);
        }
        if (decimalNumber == "")
        {
            decimalNumber = "0";
        }
        
        return wholeNumber + "." + decimalNumber;
    }

    // Single Digit Adder

    /**
     * This method adds two numbers and returns the right most digit
     * if the sum is 2 digit.
     * 
     * @param firstDigit  first user input
     * @param secondDigit second user input
     * @param carryIn     true or false, add 1 if true
     * 
     * @return the rightmost digit
     */

    public static char addDigits(char firstDigit, char secondDigit, boolean carryIn)
    {
        int firstNumber = firstDigit - '0';
        int secondNumber = secondDigit - '0';
        int carry = 0;

        if (carryIn)
        {
            carry = 1;
        }

        int sum = firstNumber + secondNumber + carry;
        return (char) (sum % 10 + '0');
    }

    /**
     * This method returns true if there is a carry, false if there isn't.
     * 
     * @param firstDigit  first user input
     * @param secondDigit second user input
     * @param carryIn     true or false
     * 
     * @return true or false
     */

    public static boolean carryOut(char firstDigit, char secondDigit, boolean carryIn)
    {
        int firstNumber = firstDigit - '0';
        int secondNumber = secondDigit - '0';
        int carry = 0;

        if (carryIn)
        {
            carry += 1;
        }

        int sum = firstNumber + secondNumber + carry;

        if (sum > 9)
        {
            return true;
        }

        return false;
    }
    
    // Calculation

    /**
     * This method adds the two numbers.
     * 
     * @param firstNumber  first user input
     * @param secondNumber second user input
     * 
     * @return the sum
     */

    public static String add(String firstNumber, String secondNumber)
    { 

        String number1 = formatResult(firstNumber);
        String number2 = formatResult(secondNumber);
        String decimal1 = extractDecimal(number1);
        String decimal2 = extractDecimal(number2);
        String whole1 = extractWholeNumber(number1);
        String whole2 = extractWholeNumber(number2);
        decimal1 = appendZeros(decimal1, decimal2.length() - decimal1.length());
        decimal2 = appendZeros(decimal2, decimal1.length() - decimal2.length());
        String sumOfDecimals = "";
        whole1 = prependZeros(whole1, whole2.length() - whole1.length());
        whole2 = prependZeros(whole2, whole1.length() - whole2.length());
        String sumOfWholes = "";
        boolean carry = false;

        for (int i = decimal1.length() - 1; i >= 0; i--)
        {
            char digit1 = decimal1.charAt(i);
            char digit2 = decimal2.charAt(i);
            char currentDigit = addDigits(digit1, digit2, carry);

            carry = carryOut(digit1, digit2, carry);

            sumOfDecimals = currentDigit + sumOfDecimals;
        }

        for (int i = whole1.length() - 1; i >= 0; i--)
        {
            char num1 = whole1.charAt(i);
            char num2 = whole2.charAt(i);
            char currentWhole = addDigits(num1, num2, carry);
            
            carry = carryOut(num1, num2, carry);

            sumOfWholes = currentWhole + sumOfWholes;   
        }

        String result = sumOfWholes + "." + sumOfDecimals;
        
        if (carry)
        {
            result = "1" + result;
        }
        
        result = formatResult(result);

        return result;
    }

    /**
     * This method multiplies the two numbers.
     * 
     * @param firstNumber  first user input
     * @param secondNumber second user input
     * 
     * @return the product
     */

    public static String multiply(String number, int numTimes)
    {
        String result = "0.0";

        if (numTimes < 0)
        {
            return number;
        }
        for (int i = 0; i < numTimes; i++)
        {
            result = add(number, result);
        }

        return result;
    }
}

