
public class ExceptionTester {
	public static void runALetterTest(String testString) throws NoALetterException
	{
		Boolean isA = false;
		try
		{
			for(int i=0; i<testString.length(); i++)
			{
				if(testString.charAt(i) == 'A') isA = true;
			}
			if(!isA) throw new NoALetterException("No A letter found!");
		}
		catch(NoALetterException e)
		{
			System.out.println(e.getMessage());
		}
	}
	
	public static void runDivisableByTwoTest(int number) throws NotDivisableByTwoException
	{
		try
		{
			if(number % 2 != 0) throw new NotDivisableByTwoException("This number is not divisable by two!");
		}
		catch(NotDivisableByTwoException e)
		{
			System.out.println(e.getMessage());
		}
	}
	
	public static int runStringTest() throws StringIndexOutOfBoundsException
	{
		String testString = new String("lol");
		char c;
		try
		{
			if(testString.length() < 4) throw new Exception("something went wrong");
			c = testString.charAt(5);
		}
		catch(StringIndexOutOfBoundsException e)
		{
			System.out.println("String exception caught!");
		}
		catch(Exception e)
		{
			System.out.println(e.getMessage());
		}
		finally
		{
			System.out.println("in finally");
			return 0;
		}
	}
	
	public static void runDivisorTest() throws ArithmeticException
	{
		int a = 5, b = 0;
		int c;
		try
		{
			c = a/b;
		}
		catch(ArithmeticException e)
		{
			System.out.println("Zero divisor exception caught!");
		}
	}
}
