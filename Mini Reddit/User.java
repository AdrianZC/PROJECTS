/*
Name: Adrian Zhu Chou
Email: azhuchou@ucsd.edu
PID: A16361462
Sources used: javatpoint, w3schools, tutors

This file is used to represent the user in our
simplified Reddit.
 */
import java.util.ArrayList;

/**
 * This class creates users, by giving them a username,
 * reputation (karma), their own posts, and posts that they
 * have upvoted and downvoted.
 * 
 * Instance variables:
 * karma: the reputation of the user
 * username: the username of the user
 * posts: the posts that the user has made
 * upvoted: the posts that the user has upvoted
 * downvoted: the posts that the user has downvoted
 */
public class User {

    /*
     * the total number of upvotes - the total number of downvotes a user has
     */
    private int karma;

    /*
     * the username of the User
     */
    private String username;

    /*
     * a list of posts this User has made, including original posts and comments
     */
    private ArrayList<Post> posts;

    /*
     * a list of other users' posts that this User upvoted
     */
    private ArrayList<Post> upvoted;

    /*
     * a list of other users' posts that this User downvoted
     */
    private ArrayList<Post> downvoted;

    /*
     * the constructor for a User
     * 
     * @param username the username of the User 
     */
    public User(String username)
    {
        this.username = username;
        this.karma = 0;
        this.posts = new ArrayList<>();
        this.upvoted = new ArrayList<>();
        this.downvoted = new ArrayList<>();   
    }

    /*
     * add a Post to the of this User's posts
     * 
     * @param post the post to add to this User's posts
     */
    public void addPost(Post post)
    {
        if (post != null)
        {
            this.posts.add(post);
            updateKarma();
        }
    }

    /*
     * update this User's krma by going through the User's posts
     * and summing up upvoteCount - downvoteCount for each post
     */
    public void updateKarma()
    {
        this.karma = 0;

        for (int i = 0; i < this.posts.size(); i++)
        {
            Post post = this.posts.get(i);
            this.karma += post.getUpvoteCount() - post.getDownvoteCount();
        }
    }

    /*
     * return the current value of karma
     * 
     * @return the current value of karma
     */
    public int getKarma()
    {
        return this.karma;
    }

    /*
     * add post to upvoted, update posts's upvoteCount accordingly
     * update the author's karma value
     */
    public void upvote(Post post)
    {
        if (post == null || this.upvoted.contains(post) || post.getAuthor() == this)
        {
            return;
        }
        if (this.downvoted.contains(post))
        {
            this.downvoted.remove(post);
            post.updateDownvoteCount(false);
        }

        this.upvoted.add(post);
        post.updateUpvoteCount(true);
        post.getAuthor().updateKarma();
    }

    /*
     * add post to downvoted, update posts's downvoteCount accordingly
     * update the author's karma value
     */
    public void downvote(Post post)
    {
        if (post == null || this.downvoted.contains(post) || post.getAuthor() == this)
        {
            return;
        }
        if (this.upvoted.contains(post))
        {
            this.upvoted.remove(post);
            post.updateUpvoteCount(false);
        }

        this.downvoted.add(post);
        post.updateDownvoteCount(true);
        post.getAuthor().updateKarma();
    }

    /*
     * Return the top post determined by the greatest
     * (upvoteCount - downvoteCount) value
     * 
     * @return the top post determined by the greatest value
     */
    public Post getTopPost()
    {
        Post topPost = null;
        int topPostScore = 0;

        if (this.posts.isEmpty())
        {
            return topPost;
        }
        
        topPost = this.posts.get(0);
        topPostScore = topPost.getUpvoteCount() - topPost.getDownvoteCount();
        
        for (int i = 1; i < this.posts.size(); i++)
        {
            Post post = this.posts.get(i);
            int postScore = post.getUpvoteCount() - post.getDownvoteCount();
            
            if (postScore > topPostScore)
            {
                topPost = post;
                topPostScore = postScore;
            }
        }

        return topPost;
    }


    /*
     * Return the top comment determined by the greatest
     * (upvoteCount - downvoteCount) value
     * 
     * @return the top comment determined by the greatest value
     */
    public Post getTopComment()
    {
        Post topComment = null;
        int topCommentScore = 0;

        if (this.posts.isEmpty())
        {
            return topComment;
        }

        topComment = this.posts.get(0);
        topCommentScore = topComment.getUpvoteCount() - topComment.getDownvoteCount();

        for (int i = 1; i < this.posts.size(); i++)
        {
            Post post = this.posts.get(i);
            int postScore = post.getUpvoteCount() - post.getDownvoteCount();
            
            if (postScore > topCommentScore)
            {
                topComment = post;
                topCommentScore = postScore;
            }
        }
        
        return topComment;
    }

    /*
     * return the list of posts (original posts and comments) made by the User
     * 
     * @return the list of posts (original posts and comments) made by the User
     */
    public ArrayList<Post> getPosts()
    {
        return this.posts;
    }

    /*
     * return a String represtnation of this User
     * 
     * @return a String representation of this User
     */
    public String toString()
    {
        return String.format("u/%s Karma: %d", this.username, this.karma);
    }
}
