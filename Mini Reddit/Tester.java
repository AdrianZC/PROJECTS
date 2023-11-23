/*
Name: Adrian Zhu Chou
Email: azhuchou@ucsd.edu
PID: A16361462
Sources used: javatpoint, w3schools, tutors

This file is used to test Post and User.
*/

 /**
  * This class tests Post and User.
  */
public class Tester {
    public static void main(String[] args) {
        User u1 = new User("CSE11Student");
        Post p1 = new Post("Title", "Content", u1);
        Post c1 = new Post("Comment", p1, u1);
        System.out.println(u1);
        System.out.println(p1);
        System.out.println(c1);
        u1.addPost(p1);
        u1.addPost(c1);
        System.out.println(u1.getTopPost());
        System.out.println(u1.getTopComment());

        User user1 = new User("Adrian");
        User user2 = new User("Pamela");

        Post post1 = new Post("Title 1", "Content 1", user1);
        Post post2 = new Post("Title 2", "Content 2", user2);
        Post comment1 = new Post("Comment 1", post1, user2);
        Post comment2 = new Post("Comment 2", post1, user1);
        Post comment3 = new Post("Comment 3", post2, user1);

        user1.addPost(post1);
        user2.addPost(post2);
        user2.addPost(comment1);
        user1.addPost(comment2);
        user1.addPost(comment3);

        user1.upvote(post2);
        user2.upvote(post1);
        user2.downvote(comment2);
        user1.downvote(comment3);

        post1.updateUpvoteCount(true);
        comment1.updateUpvoteCount(true);
        post2.updateDownvoteCount(true);
        comment3.updateDownvoteCount(true);

        System.out.println("Adrian's Top Post: " + user1.getTopPost());
        System.out.println("Pamela's Top Post: " + user2.getTopPost());
        System.out.println("Adrian's Top Comment: " + user1.getTopComment());
        System.out.println("Pamela's Top Comment: " + user2.getTopComment());
        System.out.println("Adrian's Karma: " + user1.getKarma());
        System.out.println("Pamela's Karma: " + user2.getKarma());

        System.out.println("Post 1 Title: " + post1.getTitle());
        System.out.println("Comment 1 Reply To: " + comment1.getReplyTo());
        System.out.println("Post 2 Author: " + post2.getAuthor());
        System.out.println("Post 1 Upvotes: " + post1.getUpvoteCount());
        System.out.println("Comment 3 Downvotes: " + comment3.getDownvoteCount());
        System.out.println("Comment 1 Upvotes: " + comment1.getUpvoteCount());
        System.out.println("Post 2 Downvotes: " + post2.getDownvoteCount());
    }
}