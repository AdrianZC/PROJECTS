/* 
 * Name: Adrian Zhu Chou
 * Email: azhuchou@ucsd.edu
 * PID: A16361462
 * Sources used: Textbook
 * Put "None" if you did not have any external help
   Some example of sources used would be Tutors, Textbook, and Lecture Slides
 */

 //imports
import java.util.Scanner;

/**
 * This class takes project assignment scores, midterm score, final score,
 * and calculates the grade that the student would get in the class.
 */
public class GradeCalculator 
{
    public static void main (String [] args)
    {
        Scanner input = new Scanner(System.in);

        //input pa scores
        int paScores = input.nextInt();
            if (paScores <= 0)
            {
                System.out.println("invalid input");
                return;
            }
        
        //storing pa
        int pas = 0;

        //read the pa scores
        for (int i = 0; i < paScores; i++)
        {
            int paScore = input.nextInt();//input actual pa scores
            pas += paScore;//adding actual pa scores
            if (paScore < 0 || paScore > 100)//check for invalid inputs in actual pa scores
            {
                System.out.println("invalid input");
                return;
            }
        }
        
        //input midterm score
        int midtermScore = input.nextInt();

        //input final score
        int finalScore = input.nextInt();

        input.close();

        //check for invalid inputs
        if (midtermScore < 0 || midtermScore > 100 || finalScore < 0 || finalScore > 100)
        {
            System.out.println("invalid input");
            return;
        }

        //compute the overall score
        double overallScore = ((double) pas / paScores) * 0.5 + (midtermScore * 0.125) + (finalScore * 0.375);
        System.out.println(overallScore);

        //output the letter grade based on the overall score
        char letterGrade = (overallScore >= 90 && overallScore <= 100) ? 'A' :
        (overallScore >= 80 && overallScore < 90) ? 'B' :
        (overallScore >= 70 && overallScore < 80) ? 'C' :
        (overallScore >= 60 && overallScore < 70) ? 'D' : 'F' ;
        System.out.println(letterGrade);
    }
}