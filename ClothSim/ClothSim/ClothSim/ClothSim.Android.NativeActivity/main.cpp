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

	error = System::GetInstance()->Initialize(state);

	if (error != CS_ERR_NONE)
	{
		LOGW("\nSystem.Initialize exited with error code %d\n", error);
		return;
	}

	error = System::GetInstance()->Run();

	if (error != CS_ERR_NONE)
	{
		LOGW("\nSystem.Run exited with error code %d\n", error);
		return;
	}

	error = System::GetInstance()->Shutdown();

	if (error != CS_ERR_NONE)
	{
		LOGW("\nSystem.Shutdown exited with error code %d\n", error);
		return;
	}

	System::DestroyInstance();

	LOGI("\nProgram terminated successfully.\n");
}
