/*
Name: Adrian Zhu Chou
Email: azhuchou@ucsd.edu
PID: A16361462
Sources used: javatpoint, w3schools, tutors

This file is used to create a post or comment in our
simplified Reddit.
 */
import java.util.ArrayList;

/**
 * This class creates posts, by giving them a title,
 * author, content, upvote, downvote and comments.
 * 
 * Instance variables:
 * title: the title of the post
 * content: the content of the post
 * replyTo: the post this post is replying to
 * author: the author of the post
 * upvoteCount: the number of upvotes of the post
 * downvoteCount: the number of downvotes of the post
 */
public class Post
{
    
    /*
     * the title of a Reddit post
     * if the post is a comment, then title should be null
     * if the post is an original post, then title should be non-null
     */
    private String title;

    /*
     * the content of a Reddit post
     * if the post is a comment, content is the comment a user made
     */
    private String content;
    
    /*
     * the original post or comment this Post is replying to
     * if this Post is an original post, replyTo should be null
     * replyTo should be non-null if this post is a comment
     */
    private Post replyTo;

    /*
     * the author of this Post
     */
    private User author;

    /*
     * the number of upvotes of this Post
     */
    private int upvoteCount;

    /*
     * the number of downvotes of this Post
     */
    private int downvoteCount;

    /*
     * the constructor for initializing an original post
     * 
     * @param title the title of the post
     * @param content the content of the post
     * @param author the author of the post
     */
    public Post(String title, String content, User author)
    {
        this.title = title;
        this.content = content;
        this.replyTo = null;
        this.author = author;
        this.upvoteCount = 1;
        this.downvoteCount = 0;
    }

    /*
     * the constructor for initializing a comment
     * 
     * @param content the content of the comment
     * @param replyTo the post this comment is replying to
     * @param author the author of the comment
     */
    public Post(String content, Post replyTo, User author)
    {
        this.title = null;
        this.content = content;
        this.replyTo = replyTo;
        this.author = author;
        this.upvoteCount = 1;
        this.downvoteCount = 0;
    }

    /*
     * returns the title of this Post
     * 
     * @return the title of this Post
     */
    public String getTitle()
    {
        return this.title;
    }

    /*
     * returns the Post that this Post is replying to
     * 
     * @return the Post that this Post is replying to
     */
    public Post getReplyTo()
    {
        return this.replyTo;
    }

    /*
     * returns the author of this Post
     * 
     * @return the author of this Post
     */
    public User getAuthor()
    {
        return this.author;
    }

    /*
     * return the number of upvotes of this Post
     * 
     * @return the number of upvotes of this Post
     */
    public int getUpvoteCount()
    {
        return this.upvoteCount;
    }

    /*
     * return the number of downvotes of this Post
     * 
     * @return the number of downvotes of this Post
     */
    public int getDownvoteCount()
    {
        return this.downvoteCount;
    }

    /*
     * Increment upvoteCount by 1 if isIncrement is true,
     * otherwise decrement upvoteCount by 1
     * 
     * @param isIncrement true if upvoteCount should be incremented,
     * false if upvoteCount should be decremented
     */
    public void updateUpvoteCount(boolean isIncrement)
    {
        if (isIncrement)
        {
            this.upvoteCount++;
        }
        else
        {
            this.upvoteCount--;
        }
    }

    /*
     * Increment downvoteCount by 1 if isIncrement is true,
     * otherwise decrement downvoteCount by 1
     * 
     * @param isIncrement true if downvoteCount should be incremented,
     * false if downvoteCount should be decremented
     */
    public void updateDownvoteCount(boolean isIncrement)
    {
        if (isIncrement)
        {
            this.downvoteCount++;
        }
        else
        {
            this.downvoteCount--;
        }
    }

    /*
     * Return a list of posts in the current thread,
     * starting with the original post of this post and ending with this post
     * 
     * @return a list of posts in the current thread
     */
    public ArrayList<Post> getThread()
    {
        ArrayList<Post> thread = new ArrayList<>();
        Post current = this;

        while (current != null)
        {
            thread.add(0, current);
            current = current.getReplyTo();
        }

        return thread;
    }

    /*
     * Return a string representation of this Post
     * 
     * @return a string representation of this Post
     */
    public String toString()
    {
        if (this.title != null)
        {
            return String.format("[%d|%d]\t%s\n\t%s", this.upvoteCount, this.downvoteCount, this.title, this.content);
        }
        else
        {
            return String.format("[%d|%d]\t%s", this.upvoteCount, this.downvoteCount, this.content);
        }
    }
}