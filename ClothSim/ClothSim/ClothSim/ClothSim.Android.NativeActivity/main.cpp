#include "Common.h"
#include "System.h"


/**
* This is the main entry point of a native application that is using
* android_native_app_glue.  It runs in its own thread, with its own
* event loop for receiving input events and doing other things.
*/
void android_main(android_app* state)
{
	int error;

	error = System::GetInstance()->InitAndroid(state);

	// wait for app to init
	while (!System::GetInstance()->GetRunning());

	System::GetInstance()->Run();

	System::DestroyInstance();

	LOGI("\nProgram terminated successfully.\n");
}
