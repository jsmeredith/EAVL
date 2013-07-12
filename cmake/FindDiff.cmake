#*********************************************************#
#*  File: FindDiff.cmake                                 *
#*
#*  Copyright (C) 2002-2012 The PixelLight Team (http://www.pixellight.org/)
#*
#*  This file is part of PixelLight.
#*
#*  PixelLight is free software: you can redistribute it and/or modify
#*  it under the terms of the GNU Lesser General Public License as published by
#*  the Free Software Foundation, either version 3 of the License, or
#*  (at your option) any later version.
#*
#*  PixelLight is distributed in the hope that it will be useful,
#*  but WITHOUT ANY WARRANTY; without even the implied warranty of
#*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#*  GNU Lesser General Public License for more details.
#*
#*  You should have received a copy of the GNU Lesser General Public License
#*  along with PixelLight. If not, see <http://www.gnu.org/licenses/>.
#*********************************************************#


# Find diff and patch executables
#
# Input variables:
#   SCP_FIND_REQUIRED - set this if configuration should fail without scp
#
# Output variables:
#
#   DIFF_FOUND       - set if diff was found
#   DIFF_EXECUTABLE  - path to diff executable
#   PATCH_FOUND      - set if patch was found
#   PATCH_EXECUTABLE - path to patch executable

# Search for diff
set(DIFF_EXECUTABLE)
find_program(DIFF_EXECUTABLE NAMES diff diff.exe PATHS ${CMAKE_SOURCE_DIR}/cmake/UsedTools/diff)
if(DIFF_EXECUTABLE)
	set(DIFF_FOUND ON)
else()
	set(DIFF_FOUND OFF)
endif()

# Check if diff has been found
message(STATUS "Looking for diff...")
if(DIFF_FOUND)
	message(STATUS "Looking for diff... - found ${DIFF_EXECUTABLE}")
else()
	message(STATUS "Looking for diff... - NOT found")
endif()

# Search for patch
set(PATCH_EXECUTABLE)
find_program(PATCH_EXECUTABLE NAMES patch patch.exe PATHS ${CMAKE_SOURCE_DIR}/cmake/UsedTools/diff)
if(PATCH_EXECUTABLE)
	set(PATCH_FOUND ON)
else()
	set(PATCH_FOUND OFF)
endif()

# Check if patch has been found
message(STATUS "Looking for patch...")
if(PATCH_FOUND)
	message(STATUS "Looking for patch... - found ${PATCH_EXECUTABLE}")
else()
	message(STATUS "Looking for patch... - NOT found")
endif()

# Mark variables as advanced
mark_as_advanced(DIFF_EXECUTABLE DIFF_FOUND PATCH_EXECUTABLE PATCH_FOUND)
